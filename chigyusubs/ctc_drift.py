"""CTC drift detection + repair.

Operates on `ctc_words.json` (output of align_ctc.py). Detects segments where
forced alignment placed tokens across silence — pre-roll, BGM-masked passages,
domain-mismatch pockets — and repairs by clustering words on intra-word gaps,
validating clusters against VAD, and re-timing the segment to its real-cluster
span.

See `docs/timing-architecture-2026-05.md` for grounded thresholds and
research citations.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field


# Defaults grounded in research (see docs/timing-architecture-2026-05.md).
DEFAULT_CLUSTER_GAP_S = 0.5          # WhisperX/stable-ts inter-word gap consensus
DEFAULT_MAX_PRE_SPEECH_LEAD_S = 0.08  # Netflix 1-2 frames @24fps
DEFAULT_MIN_CUE_S = 0.5              # Netflix min cue duration
DEFAULT_TARGET_JP_CPS = 4.0          # Netflix Japanese regular
DEFAULT_GHOST_VAD_OVERLAP_S = 0.1    # min VAD overlap to count cluster as real
DEFAULT_VAD_BRIDGE_S = 0.05          # min VAD speech inside a gap to treat the gap as natural pause
DEFAULT_NEAR_VAD_LOOKAHEAD_S = 1.0   # tolerance for "cluster start near VAD onset"
DEFAULT_MIN_INTRA_GAP_S = 3.0        # only fire when at least one >=3s gap inside seg —
                                     # blank-collapse drift signature, well past natural pause


@dataclass
class VadRegion:
    start: float
    end: float


@dataclass
class Cluster:
    words: list[dict]
    real: bool = False
    vad_overlap_s: float = 0.0

    @property
    def start(self) -> float:
        return float(self.words[0]["start"])

    @property
    def end(self) -> float:
        return float(self.words[-1]["end"])

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def char_count(self) -> int:
        return sum(len(w.get("word", "")) for w in self.words)


@dataclass
class SegmentDiag:
    seg_idx: int
    text: str
    old_start: float
    old_end: float
    new_start: float
    new_end: float
    n_clusters: int
    n_real_clusters: int
    n_ghost_words_redistributed: int
    max_intra_gap_s: float
    action: str  # "kept", "trimmed", "relocated", "blocked"
    note: str = ""

    @property
    def shift_s(self) -> float:
        return self.new_start - self.old_start


def vad_overlap_s(start: float, end: float, regions: list[VadRegion]) -> float:
    if end <= start or not regions:
        return 0.0
    ends = [r.end for r in regions]
    lo = bisect_left(ends, start)
    total = 0.0
    for r in regions[lo:]:
        if r.start >= end:
            break
        total += min(end, r.end) - max(start, r.start)
    return max(0.0, total)


def first_vad_onset_in(start: float, end: float, regions: list[VadRegion]) -> float | None:
    """First region.start in [start, end), or None."""
    starts = [r.start for r in regions]
    lo = bisect_left(starts, start)
    if lo >= len(regions):
        return None
    if regions[lo].start >= end:
        return None
    return regions[lo].start


def cluster_words(
    words: list[dict],
    gap_threshold: float,
    vad: list[VadRegion] | None = None,
    vad_bridge_s: float = DEFAULT_VAD_BRIDGE_S,
) -> list[Cluster]:
    """Group consecutive words into clusters, splitting only on silent gaps.

    A gap >= gap_threshold splits a cluster only if it falls in VAD silence
    (less than vad_bridge_s of VAD speech inside the gap). Gaps that contain
    real speech are natural pauses and don't split the cluster — this prevents
    misclassifying breath/hesitation pauses as drift.
    """
    if not words:
        return []
    clusters: list[list[dict]] = [[words[0]]]
    for i in range(1, len(words)):
        gap_start = float(words[i - 1]["end"])
        gap_end = float(words[i]["start"])
        gap = gap_end - gap_start
        split = False
        if gap >= gap_threshold:
            if vad is None:
                split = True
            else:
                bridge = vad_overlap_s(gap_start, gap_end, vad)
                split = bridge < vad_bridge_s
        if split:
            clusters.append([words[i]])
        else:
            clusters[-1].append(words[i])
    return [Cluster(words=c) for c in clusters]


def score_clusters(
    clusters: list[Cluster],
    vad: list[VadRegion],
    *,
    ghost_vad_overlap_s: float,
):
    """Mark cluster real iff it has at least ghost_vad_overlap_s of VAD coverage.

    No words-fallback: a cluster of 3 words sitting in pure silence is still
    drift, not legitimate speech. The right way to recover such a cluster is
    a stronger VAD model, not a heuristic that ignores VAD.
    """
    for c in clusters:
        c.vad_overlap_s = vad_overlap_s(c.start, c.end, vad)
        c.real = c.vad_overlap_s >= ghost_vad_overlap_s


def _redistribute_word_times(words: list[dict], new_start: float, new_end: float) -> list[dict]:
    """Spread word timestamps uniformly across [new_start, new_end] by character weight.

    Returns new word dicts; original list untouched.
    """
    if not words:
        return []
    n = len(words)
    span = max(0.001, new_end - new_start)
    # Weight by character count of each word so multi-char words get more room.
    weights = [max(1, len(w.get("word", ""))) for w in words]
    total_w = sum(weights)
    out: list[dict] = []
    cursor = new_start
    for w, weight in zip(words, weights):
        w_dur = span * (weight / total_w)
        new_w = dict(w)
        new_w["start"] = round(cursor, 3)
        new_w["end"] = round(cursor + w_dur, 3)
        out.append(new_w)
        cursor += w_dur
    # Force last word.end = new_end exactly.
    if out:
        out[-1]["end"] = round(new_end, 3)
    return out


def _segment_text_chars(seg: dict) -> int:
    text = (seg.get("text") or "").strip()
    return len(text)


def _is_drift_suspect(max_intra_gap_s: float, min_intra_gap_s: float) -> bool:
    """Gross-drift signature: at least one >=min_intra_gap_s gap between consecutive words.

    A 3s+ gap inside a Japanese cue can't be natural pause — it's the
    blank-collapse pattern where CTC sprinkles tokens across silence. Sub-3s
    intra-word spacing (including multi-word seg with 0.5-1s pauses) is left
    alone: relocating those is more likely to introduce regressions than help,
    given Silero/Whisper both have their own timing errors at that scale.

    Empirical (san-nomi 2026-05): seg 0 (3.84s gap) and seg 17 (5.72s gap)
    fire under this rule — both confirmed gross drift. Seg 128 (0s), seg 220
    (0.82s), seg 753 (0.4s) don't fire — all confirmed non-fixable or
    regression-prone.
    """
    return max_intra_gap_s >= min_intra_gap_s


def repair_segments(
    segments: list[dict],
    vad: list[VadRegion],
    *,
    cluster_gap_s: float = DEFAULT_CLUSTER_GAP_S,
    max_pre_speech_lead_s: float = DEFAULT_MAX_PRE_SPEECH_LEAD_S,
    min_cue_s: float = DEFAULT_MIN_CUE_S,
    target_jp_cps: float = DEFAULT_TARGET_JP_CPS,
    ghost_vad_overlap_s: float = DEFAULT_GHOST_VAD_OVERLAP_S,
    vad_bridge_s: float = DEFAULT_VAD_BRIDGE_S,
    near_vad_lookahead_s: float = DEFAULT_NEAR_VAD_LOOKAHEAD_S,
    min_intra_gap_s: float = DEFAULT_MIN_INTRA_GAP_S,
) -> tuple[list[dict], list[SegmentDiag]]:
    """Return (corrected_segments, diagnostics).

    Segments are processed in order; a `cursor` tracks the last-confirmed end
    so relocations don't push past it. Original segments are not mutated.
    """
    out: list[dict] = []
    diags: list[SegmentDiag] = []
    cursor = 0.0  # last end of confirmed-good content

    for i, seg in enumerate(segments):
        seg = dict(seg)  # shallow copy
        words = list(seg.get("words") or [])
        old_start = float(seg.get("start", 0.0))
        old_end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()

        # Empty / zero-duration text-only segments: pass through.
        if not words:
            out.append(seg)
            cursor = max(cursor, old_end)
            diags.append(SegmentDiag(
                seg_idx=i, text=text, old_start=old_start, old_end=old_end,
                new_start=old_start, new_end=old_end,
                n_clusters=0, n_real_clusters=0, n_ghost_words_redistributed=0,
                max_intra_gap_s=0.0, action="kept", note="no words",
            ))
            continue

        clusters = cluster_words(words, cluster_gap_s, vad=vad, vad_bridge_s=vad_bridge_s)
        score_clusters(clusters, vad, ghost_vad_overlap_s=ghost_vad_overlap_s)

        # Compute max intra-word gap (largest gap between consecutive words within seg)
        max_gap = 0.0
        for j in range(1, len(words)):
            g = float(words[j]["start"]) - float(words[j - 1]["end"])
            if g > max_gap:
                max_gap = g

        suspect = _is_drift_suspect(max_gap, min_intra_gap_s)

        if not suspect:
            out.append(seg)
            cursor = max(cursor, old_end)
            diags.append(SegmentDiag(
                seg_idx=i, text=text, old_start=old_start, old_end=old_end,
                new_start=old_start, new_end=old_end,
                n_clusters=len(clusters),
                n_real_clusters=sum(1 for c in clusters if c.real),
                n_ghost_words_redistributed=0,
                max_intra_gap_s=round(max_gap, 3),
                action="kept", note="not suspect",
            ))
            continue

        real_clusters = [c for c in clusters if c.real]
        ghost_clusters = [c for c in clusters if not c.real]

        # Bound for relocation: don't push past next segment's start.
        next_seg_start = float(segments[i + 1]["start"]) if i + 1 < len(segments) else float("inf")
        # Lower bound: don't overlap previous segment.
        prev_end = cursor

        if real_clusters:
            # Mixed (or all-real): retime to real-cluster span. Keep ALL words —
            # ghost-cluster words may be real audio the wav2vec2 model couldn't
            # anchor (romaji, numerals, loanwords, low-confidence regions). The
            # cue-level signal (this region drifted) is robust; per-word ghost
            # vs real classification is too brittle to justify dropping text.
            new_start = max(prev_end, real_clusters[0].start)
            new_end = min(next_seg_start, real_clusters[-1].end)
            if new_end - new_start < 0.05:
                new_end = min(next_seg_start, new_start + min_cue_s)
            ghost_count = sum(len(c.words) for c in ghost_clusters)
            unchanged = (
                not ghost_clusters
                and abs(new_start - old_start) < 1e-3
                and abs(new_end - old_end) < 1e-3
            )
            if unchanged:
                # No retiming needed; preserve original CTC word timestamps.
                new_words = words
            elif not ghost_clusters:
                # Span shrunk slightly (e.g. monotonic clamp); preserve real
                # CTC times and only nudge the boundary words.
                new_words = list(words)
            else:
                # Ghost cluster present — retime everything across new span by
                # character weight so ghost-cluster words sit in transcript order.
                new_words = _redistribute_word_times(words, new_start, new_end)
            seg["start"] = round(new_start, 3)
            seg["end"] = round(new_end, 3)
            seg["words"] = new_words
            out.append(seg)
            cursor = max(cursor, new_end)
            diags.append(SegmentDiag(
                seg_idx=i, text=text, old_start=old_start, old_end=old_end,
                new_start=new_start, new_end=new_end,
                n_clusters=len(clusters),
                n_real_clusters=len(real_clusters),
                n_ghost_words_redistributed=ghost_count,
                max_intra_gap_s=round(max_gap, 3),
                action="kept" if unchanged else "trimmed",
                note=("all clusters real" if unchanged
                      else f"kept {len(real_clusters)}/{len(clusters)} clusters; ghost words retimed"),
            ))
            continue

        # All ghost: relocate.
        chars = _segment_text_chars(seg)
        target_dur = max(min_cue_s, chars / target_jp_cps if chars else min_cue_s)

        # Search VAD onsets in (prev_end, next_seg_start). Prefer ones near old_start.
        search_lo = prev_end + 0.05
        search_hi = next_seg_start - 0.05
        candidate_onset = None
        if search_hi > search_lo:
            # Prefer earliest VAD onset >= search_lo within range.
            candidate_onset = first_vad_onset_in(search_lo, search_hi, vad)

        if candidate_onset is not None:
            new_start = max(search_lo, candidate_onset - max_pre_speech_lead_s)
            new_end = min(search_hi, new_start + target_dur)
            if new_end - new_start < min_cue_s and search_hi - search_lo >= min_cue_s:
                new_end = min(search_hi, new_start + min_cue_s)
            action = "relocated"
            note = f"vad onset {candidate_onset:.3f}"
        elif search_hi - search_lo >= min_cue_s:
            # No VAD anchor; pack against next segment's start.
            new_end = search_hi
            new_start = max(search_lo, new_end - target_dur)
            action = "relocated"
            note = "no vad anchor; packed before next seg"
        else:
            # No room. Leave alone.
            out.append(seg)
            cursor = max(cursor, old_end)
            diags.append(SegmentDiag(
                seg_idx=i, text=text, old_start=old_start, old_end=old_end,
                new_start=old_start, new_end=old_end,
                n_clusters=len(clusters), n_real_clusters=0,
                n_ghost_words_redistributed=0,
                max_intra_gap_s=round(max_gap, 3),
                action="blocked", note="no room to relocate",
            ))
            continue

        new_words = _redistribute_word_times(words, new_start, new_end)
        seg["start"] = round(new_start, 3)
        seg["end"] = round(new_end, 3)
        seg["words"] = new_words
        out.append(seg)
        cursor = max(cursor, new_end)
        diags.append(SegmentDiag(
            seg_idx=i, text=text, old_start=old_start, old_end=old_end,
            new_start=new_start, new_end=new_end,
            n_clusters=len(clusters),
            n_real_clusters=0,
            n_ghost_words_redistributed=len(words),
            max_intra_gap_s=round(max_gap, 3),
            action=action, note=note,
        ))

    return out, diags


def summarize(diags: list[SegmentDiag]) -> dict:
    counts: dict[str, int] = {}
    shifts: list[float] = []
    ghost_words = 0
    for d in diags:
        counts[d.action] = counts.get(d.action, 0) + 1
        if d.action in ("trimmed", "relocated"):
            shifts.append(d.shift_s)
        ghost_words += d.n_ghost_words_redistributed
    out = {"counts": counts, "ghost_words_redistributed": ghost_words}
    if shifts:
        abs_shifts = [abs(s) for s in shifts]
        out.update({
            "n_modified": len(shifts),
            "max_abs_shift_s": round(max(abs_shifts), 3),
            "median_abs_shift_s": round(sorted(abs_shifts)[len(abs_shifts) // 2], 3),
            "mean_abs_shift_s": round(sum(abs_shifts) / len(abs_shifts), 3),
        })
    else:
        out["n_modified"] = 0
    return out
