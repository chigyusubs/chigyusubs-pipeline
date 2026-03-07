"""Semantic reflow: re-segment word-level timestamps into subtitle cues."""

from __future__ import annotations

import re
from bisect import bisect_left


_ZERO_CLUSTER_EPSILON_S = 0.05
_INTERJECTION_MAX_CHARS = 8
_TARGET_CPS = 14.0
_MAX_CPS = 20.0
_SENTENCE_END_RE = re.compile(r"[。！？!?」』）】]$")
_COMMA_RE = re.compile(r"[、,]$")


def reflow_words(
    segments: list[dict],
    pause_threshold: float = 0.3,
    max_cue_s: float = 10.0,
    min_cue_s: float = 0.3,
    target_cps: float = _TARGET_CPS,
    max_cps: float = _MAX_CPS,
) -> list[dict]:
    """Re-segment word timestamps into cues aligned to natural pauses.

    Args:
        segments: List of segment dicts with "words" arrays (faster-whisper format).
        pause_threshold: Gap in seconds between words that triggers a cue break.
        max_cue_s: Maximum cue duration; longer groups get split.
        min_cue_s: Minimum cue duration; shorter groups merge into the next.
        target_cps: Desired upper-bound target for chars-per-second after retiming.
        max_cps: Hard readability threshold that triggers retiming/merging.

    Returns:
        List of cue dicts with keys: start, end, text, words.
    """
    all_words = []
    zero_duration_segments = []
    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        seg_text = (seg.get("text") or "").strip()
        if seg_end <= seg_start and seg_text:
            zero_duration_segments.append({
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
            })
        for w in seg.get("words", []):
            if w.get("end", 0.0) <= w.get("start", 0.0):
                continue
            all_words.append(w)

    if not all_words:
        return []

    groups: list[list[dict]] = []
    current_group: list[dict] = [all_words[0]]

    for i in range(1, len(all_words)):
        gap = all_words[i]["start"] - all_words[i - 1]["end"]
        if gap >= pause_threshold:
            groups.append(current_group)
            current_group = [all_words[i]]
        else:
            current_group.append(all_words[i])

    if current_group:
        groups.append(current_group)

    merged: list[list[dict]] = []
    i = 0
    while i < len(groups):
        g = groups[i]
        dur = g[-1]["end"] - g[0]["start"]
        if dur < min_cue_s and i + 1 < len(groups):
            gap_to_next = groups[i + 1][0]["start"] - g[-1]["end"]
            if gap_to_next < pause_threshold * 2:
                groups[i + 1] = g + groups[i + 1]
                i += 1
                continue
        merged.append(g)
        i += 1

    final_groups: list[list[dict]] = []
    for g in merged:
        _split_group(g, max_cue_s, final_groups)

    cues = []
    for g in final_groups:
        text = "".join(w["word"] for w in g).strip()
        if not text:
            continue
        cues.append({
            "start": g[0]["start"],
            "end": g[-1]["end"],
            "text": text,
            "words": g,
        })

    if zero_duration_segments:
        _attach_zero_duration_segments(cues, zero_duration_segments)

    _normalize_cues(
        cues,
        pause_threshold=pause_threshold,
        min_cue_s=min_cue_s,
        max_cue_s=max_cue_s,
        target_cps=target_cps,
        max_cps=max_cps,
    )

    return cues


def reflow_lines(
    segments: list[dict],
    max_cue_s: float = 7.0,
    max_cue_chars: int = 45,
    max_lines: int = 2,
    min_cue_s: float = 1.0,
    target_cps: float = _TARGET_CPS,
) -> list[dict]:
    """Reflow CTC-aligned segments treating each transcript line as atomic.

    This avoids mid-word splits entirely. Lines are merged when they're short
    or close together, and split at sentence-ending punctuation when too long.

    Args:
        segments: CTC alignment output (list of segment dicts with start/end/text).
        max_cue_s: Split cues longer than this at sentence boundaries.
        max_cue_chars: Split cues with more characters than this.
        max_lines: Maximum number of lines per cue (default: 2, subtitle standard).
        min_cue_s: Merge lines shorter than this into neighbors.
        target_cps: Target CPS for boundary expansion.
    """
    # Collect timed lines, separating zero-duration
    lines: list[dict] = []
    zero_lines: list[dict] = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        entry = {"start": start, "end": end, "text": text}
        if end <= start:
            zero_lines.append(entry)
        else:
            lines.append(entry)

    if not lines:
        return []

    # Phase 1: Group adjacent lines into cues based on inter-line gaps.
    # Use a simple greedy approach: start a new cue when the gap is large
    # or when adding the next line would exceed max duration/chars.
    cues: list[dict] = []
    current_lines: list[dict] = [lines[0]]

    for i in range(1, len(lines)):
        prev = lines[i - 1]
        curr = lines[i]
        gap = curr["start"] - prev["end"]

        # Would merging exceed limits?
        merged_dur = curr["end"] - current_lines[0]["start"]
        merged_chars = sum(len(ln["text"]) for ln in current_lines) + len(curr["text"])

        # Break conditions
        if merged_dur > max_cue_s or merged_chars > max_cue_chars or len(current_lines) >= max_lines:
            cues.append(_lines_to_cue(current_lines))
            current_lines = [curr]
        elif gap > max_cue_s * 0.5:
            # Large gap — always break
            cues.append(_lines_to_cue(current_lines))
            current_lines = [curr]
        else:
            current_lines.append(curr)

    if current_lines:
        cues.append(_lines_to_cue(current_lines))

    # Phase 2: Merge short cues into neighbors
    _merge_short_cues(cues, min_cue_s=min_cue_s, max_cue_s=max_cue_s, max_cue_chars=max_cue_chars, max_lines=max_lines)

    # Phase 3: Split oversized cues at sentence-ending punctuation.
    # First try splitting at line boundaries, then within lines at punctuation.
    cues = _split_long_cues(cues, max_cue_s=max_cue_s, max_cue_chars=max_cue_chars, max_lines=max_lines)
    cues = _split_long_single_lines(cues, segments, max_cue_s=max_cue_s, max_cue_chars=max_cue_chars)

    # Phase 4: Clamp overlong cues where text is too short for the duration
    _clamp_sparse_cues(cues, max_cue_s=max_cue_s, target_cps=target_cps)

    # Phase 5: Expand short cue boundaries into neighboring silence
    _expand_cue_boundaries(cues, min_cue_s=min_cue_s, max_cue_s=max_cue_s, target_cps=target_cps)

    # Phase 6: Attach zero-duration lines
    if zero_lines:
        _attach_zero_duration_segments(cues, zero_lines)

    # Phase 7: Enforce max_lines — trim cues bloated by zero-duration attachment
    _enforce_max_lines(cues, max_lines=max_lines)

    return cues


def _lines_to_cue(lines: list[dict]) -> dict:
    text = "\n".join(ln["text"] for ln in lines)
    return {
        "start": lines[0]["start"],
        "end": lines[-1]["end"],
        "text": text,
        "lines": lines,
    }


def _cue_text_chars(cue: dict) -> int:
    return len(cue["text"].replace("\n", "").replace(" ", ""))


def _cue_line_count(cue: dict) -> int:
    return cue["text"].count("\n") + 1


def _merge_short_cues(
    cues: list[dict],
    *,
    min_cue_s: float,
    max_cue_s: float,
    max_cue_chars: int,
    max_lines: int = 2,
):
    """Merge cues that are too short into the best neighbor."""
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(cues):
            dur = cues[i]["end"] - cues[i]["start"]
            chars = _cue_text_chars(cues[i])
            if dur >= min_cue_s and chars >= 3:
                i += 1
                continue

            # Try merging with the neighbor that produces the better result
            best_j = None
            best_score = float("inf")
            for j in [i - 1, i + 1]:
                if j < 0 or j >= len(cues):
                    continue
                merged_lines = _cue_line_count(cues[i]) + _cue_line_count(cues[j])
                merged_dur = max(cues[i]["end"], cues[j]["end"]) - min(cues[i]["start"], cues[j]["start"])
                merged_chars = _cue_text_chars(cues[i]) + _cue_text_chars(cues[j])
                if merged_dur > max_cue_s or merged_chars > max_cue_chars or merged_lines > max_lines:
                    continue
                # Prefer smaller resulting cue, prefer right neighbor for fragments
                score = merged_dur + (0 if j == i + 1 else 0.1)
                if score < best_score:
                    best_score = score
                    best_j = j

            if best_j is None:
                i += 1
                continue

            left, right = (i, best_j) if i < best_j else (best_j, i)
            merged = _merge_adjacent_cues(cues[left], cues[right])
            cues[left:right + 1] = [merged]
            changed = True
            break  # restart scan


def _merge_adjacent_cues(left: dict, right: dict) -> dict:
    merged_lines = list(left.get("lines", [{"start": left["start"], "end": left["end"], "text": left["text"]}]))
    merged_lines += list(right.get("lines", [{"start": right["start"], "end": right["end"], "text": right["text"]}]))
    return {
        "start": left["start"],
        "end": right["end"],
        "text": left["text"] + "\n" + right["text"],
        "lines": merged_lines,
    }


def _split_long_cues(
    cues: list[dict],
    *,
    max_cue_s: float,
    max_cue_chars: int,
    max_lines: int = 2,
) -> list[dict]:
    """Split cues that exceed limits at sentence-ending punctuation."""
    result: list[dict] = []
    for cue in cues:
        dur = cue["end"] - cue["start"]
        chars = _cue_text_chars(cue)
        lines = cue.get("lines", [])
        needs_split = dur > max_cue_s or chars > max_cue_chars or len(lines) > max_lines

        if not needs_split or len(lines) < 2:
            result.append(cue)
            continue

        # Find best split point: prefer sentence-ending punctuation near the middle
        best_split = None
        best_score = float("inf")
        total_chars = sum(len(ln["text"]) for ln in lines)

        for split_after in range(len(lines) - 1):
            left_lines = lines[: split_after + 1]
            right_lines = lines[split_after + 1 :]
            left_chars = sum(len(ln["text"]) for ln in left_lines)
            right_chars = total_chars - left_chars
            left_dur = left_lines[-1]["end"] - left_lines[0]["start"]
            right_dur = right_lines[-1]["end"] - right_lines[0]["start"]

            # Both halves must be viable
            if left_dur < 0.3 or right_dur < 0.3:
                continue

            # Score: prefer balanced splits, bonus for sentence-ending punctuation
            balance = abs(left_chars - right_chars)
            punct_bonus = -10 if _SENTENCE_END_RE.search(left_lines[-1]["text"]) else 0
            score = balance + punct_bonus

            if score < best_score:
                best_score = score
                best_split = split_after

        if best_split is not None:
            left_cue = _lines_to_cue(lines[: best_split + 1])
            right_cue = _lines_to_cue(lines[best_split + 1 :])
            # Recursively split if still too long
            result.extend(_split_long_cues([left_cue], max_cue_s=max_cue_s, max_cue_chars=max_cue_chars, max_lines=max_lines))
            result.extend(_split_long_cues([right_cue], max_cue_s=max_cue_s, max_cue_chars=max_cue_chars, max_lines=max_lines))
        else:
            result.append(cue)

    return result


def _split_long_single_lines(
    cues: list[dict],
    segments: list[dict],
    *,
    max_cue_s: float,
    max_cue_chars: int,
) -> list[dict]:
    """Split single-line cues that exceed limits using character timestamps.

    For cues that are a single transcript line (can't be split at line
    boundaries), find sentence-ending punctuation and use the character-level
    timestamps from the original segments to split.
    """
    # Build a lookup from segment text to its word-level timestamps
    seg_by_text: dict[str, dict] = {}
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if text and text not in seg_by_text:
            seg_by_text[text] = seg

    result: list[dict] = []
    for cue in cues:
        dur = cue["end"] - cue["start"]
        chars = _cue_text_chars(cue)
        lines = cue.get("lines", [])

        if (dur <= max_cue_s and chars <= max_cue_chars) or len(lines) != 1:
            result.append(cue)
            continue

        # Single line that's too long — try to split at punctuation
        line = lines[0]
        seg = seg_by_text.get(line["text"])
        if not seg or not seg.get("words"):
            result.append(cue)
            continue

        words = seg["words"]
        # Find sentence-ending punctuation in the word list
        split_points = []
        for wi, w in enumerate(words[:-1]):  # don't split after last word
            if _SENTENCE_END_RE.search(w.get("word", "")):
                split_points.append(wi)

        # Fallback: split at comma if no sentence-ending punctuation
        if not split_points:
            for wi, w in enumerate(words[:-1]):
                if _COMMA_RE.search(w.get("word", "")):
                    split_points.append(wi)

        if not split_points:
            result.append(cue)
            continue

        # Pick the most balanced split point
        total_words = len(words)
        best = min(split_points, key=lambda sp: abs(sp - total_words / 2))

        left_words = words[: best + 1]
        right_words = words[best + 1 :]

        left_text = "".join(w["word"] for w in left_words)
        right_text = "".join(w["word"] for w in right_words)

        left_cue = {
            "start": left_words[0]["start"],
            "end": left_words[-1]["end"],
            "text": left_text,
            "lines": [{"start": left_words[0]["start"], "end": left_words[-1]["end"], "text": left_text}],
        }
        right_cue = {
            "start": right_words[0]["start"],
            "end": right_words[-1]["end"],
            "text": right_text,
            "lines": [{"start": right_words[0]["start"], "end": right_words[-1]["end"], "text": right_text}],
        }
        # Recursively split if still too long
        result.extend(_split_long_single_lines([left_cue], segments, max_cue_s=max_cue_s, max_cue_chars=max_cue_chars))
        result.extend(_split_long_single_lines([right_cue], segments, max_cue_s=max_cue_s, max_cue_chars=max_cue_chars))

    return result


def _enforce_max_lines(cues: list[dict], *, max_lines: int):
    """Trim cues that exceed max_lines, keeping only the first max_lines lines."""
    for cue in cues:
        lines = cue["text"].split("\n")
        if len(lines) <= max_lines:
            continue
        cue["text"] = "\n".join(lines[:max_lines])


def _clamp_sparse_cues(cues: list[dict], *, max_cue_s: float, target_cps: float):
    """Shrink cues where text is far too short for the aligned duration.

    This handles CTC alignment artifacts where a short utterance gets spread
    across a large audio window (e.g., 5 characters spanning 10 seconds).
    """
    for i, cue in enumerate(cues):
        dur = cue["end"] - cue["start"]
        if dur <= max_cue_s:
            continue
        chars = _cue_text_chars(cue)
        desired_dur = max(max_cue_s, chars / target_cps)
        if dur <= desired_dur:
            continue

        # Shrink to desired_dur, centered on the midpoint of the cue
        mid = (cue["start"] + cue["end"]) / 2
        # Respect neighboring cue boundaries
        left_bound = cues[i - 1]["end"] if i > 0 else 0.0
        right_bound = cues[i + 1]["start"] if i + 1 < len(cues) else float("inf")

        new_start = max(left_bound, mid - desired_dur / 2)
        new_end = min(right_bound, new_start + desired_dur)
        new_start = max(left_bound, new_end - desired_dur)

        cue["start"] = round(new_start, 3)
        cue["end"] = round(new_end, 3)


def _attach_zero_duration_segments(cues: list[dict], zero_segments: list[dict]):
    if not cues:
        return

    clusters = _cluster_zero_duration_segments(zero_segments)
    starts = [float(c["start"]) for c in cues]

    for cluster in clusters:
        target_idx = _find_target_cue_index(cues, starts, cluster)
        if target_idx is None:
            continue
        prepend = _should_prepend(cluster, cues, target_idx)
        _merge_cluster_into_cue(cues[target_idx], cluster, prepend=prepend)


def _cluster_zero_duration_segments(zero_segments: list[dict]) -> list[dict]:
    ordered = sorted(zero_segments, key=lambda s: (float(s["start"]), float(s["end"])))
    clusters: list[dict] = []
    current: dict | None = None
    for seg in ordered:
        seg_start = float(seg["start"])
        if current is None or abs(seg_start - float(current["timestamp"])) > _ZERO_CLUSTER_EPSILON_S:
            current = {
                "timestamp": seg_start,
                "segments": [seg],
            }
            clusters.append(current)
        else:
            current["segments"].append(seg)

    for cluster in clusters:
        parts = [s["text"].strip() for s in cluster["segments"] if s["text"].strip()]
        cluster["text"] = "\n".join(parts)
        cluster["segment_count"] = len(cluster["segments"])
    return clusters


def _find_target_cue_index(cues: list[dict], starts: list[float], cluster: dict) -> int | None:
    ts = float(cluster["timestamp"])
    idx = bisect_left(starts, ts)
    prev_idx = idx - 1 if idx > 0 else None
    next_idx = idx if idx < len(cues) else None

    if prev_idx is None and next_idx is None:
        return None
    if prev_idx is None:
        return next_idx
    if next_idx is None:
        return prev_idx

    prev_cue = cues[prev_idx]
    next_cue = cues[next_idx]
    prev_dist = abs(ts - float(prev_cue["end"]))
    next_dist = abs(float(next_cue["start"]) - ts)
    if prev_dist < next_dist:
        return prev_idx
    if next_dist < prev_dist:
        return next_idx
    return next_idx if _is_question_or_setup(cluster["text"]) else prev_idx


def _should_prepend(cluster: dict, cues: list[dict], target_idx: int) -> bool:
    ts = float(cluster["timestamp"])
    target = cues[target_idx]
    if abs(float(target["start"]) - ts) <= abs(ts - float(target["end"])):
        return True
    if _is_question_or_setup(cluster["text"]):
        return True
    if _is_short_interjection(cluster["text"]):
        return False
    return False


def _merge_cluster_into_cue(cue: dict, cluster: dict, prepend: bool):
    cluster_text = cluster["text"].strip()
    if not cluster_text:
        return
    cue.setdefault("repaired_zero_duration_texts", []).append({
        "timestamp": float(cluster["timestamp"]),
        "text": cluster_text,
        "segment_count": cluster["segment_count"],
    })
    if prepend:
        cue["text"] = f"{cluster_text}\n{cue['text']}".strip()
    else:
        cue["text"] = f"{cue['text']}\n{cluster_text}".strip()


def _is_short_interjection(text: str) -> bool:
    compact = text.replace(" ", "").replace("\n", "")
    return len(compact) <= _INTERJECTION_MAX_CHARS


def _is_question_or_setup(text: str) -> bool:
    stripped = text.strip()
    if stripped.endswith(("?", "？")):
        return True
    return any(token in stripped for token in ("何", "どう", "なんで", "どこ", "誰", "かい", "って"))


def _normalize_cues(
    cues: list[dict],
    *,
    pause_threshold: float,
    min_cue_s: float,
    max_cue_s: float,
    target_cps: float,
    max_cps: float,
):
    if not cues:
        return

    _expand_cue_boundaries(cues, min_cue_s=min_cue_s, max_cue_s=max_cue_s, target_cps=target_cps)

    changed = True
    while changed:
        changed = _merge_problem_cues(
            cues,
            pause_threshold=pause_threshold,
            min_cue_s=min_cue_s,
            max_cue_s=max_cue_s,
            max_cps=max_cps,
        )
        if changed:
            _expand_cue_boundaries(cues, min_cue_s=min_cue_s, max_cue_s=max_cue_s, target_cps=target_cps)


def _expand_cue_boundaries(cues: list[dict], *, min_cue_s: float, max_cue_s: float, target_cps: float):
    for i, cue in enumerate(cues):
        dur = _cue_duration(cue)
        chars = _cue_chars(cue)
        desired = max(min_cue_s, min(max_cue_s, chars / target_cps if chars else min_cue_s))
        if dur >= desired:
            continue
        extra = desired - dur
        left_gap = cue["start"] - cues[i - 1]["end"] if i > 0 else 0.0
        right_gap = cues[i + 1]["start"] - cue["end"] if i + 1 < len(cues) else 0.0
        left_take = min(left_gap, extra / 2)
        right_take = min(right_gap, extra - left_take)
        remaining = extra - left_take - right_take
        if remaining > 0 and left_gap > left_take:
            add = min(left_gap - left_take, remaining)
            left_take += add
            remaining -= add
        if remaining > 0 and right_gap > right_take:
            add = min(right_gap - right_take, remaining)
            right_take += add
        cue["start"] -= left_take
        cue["end"] += right_take


def _merge_problem_cues(
    cues: list[dict],
    *,
    pause_threshold: float,
    min_cue_s: float,
    max_cue_s: float,
    max_cps: float,
) -> bool:
    i = 0
    while i < len(cues):
        cue = cues[i]
        dur = _cue_duration(cue)
        cps = _cue_cps(cue)
        if dur >= min_cue_s and cps <= max_cps:
            i += 1
            continue

        prev_candidate = _merge_candidate(cues, i, i - 1, pause_threshold, max_cue_s, max_cps) if i > 0 else None
        next_candidate = _merge_candidate(cues, i, i + 1, pause_threshold, max_cue_s, max_cps) if i + 1 < len(cues) else None
        choice = _choose_merge_candidate(prev_candidate, next_candidate)
        if not choice:
            i += 1
            continue

        left_idx, right_idx = sorted((choice["left_idx"], choice["right_idx"]))
        merged = _merge_two_cues(cues[left_idx], cues[right_idx])
        cues[left_idx:right_idx + 1] = [merged]
        return True
    return False


def _merge_candidate(
    cues: list[dict],
    base_idx: int,
    other_idx: int,
    pause_threshold: float,
    max_cue_s: float,
    max_cps: float,
) -> dict | None:
    if other_idx < 0 or other_idx >= len(cues):
        return None
    left_idx, right_idx = sorted((base_idx, other_idx))
    left = cues[left_idx]
    right = cues[right_idx]
    gap = right["start"] - left["end"]
    if gap > pause_threshold * 2:
        return None
    merged = _merge_two_cues(left, right)
    dur = _cue_duration(merged)
    if dur > max_cue_s:
        return None
    cps = _cue_cps(merged)
    return {
        "left_idx": left_idx,
        "right_idx": right_idx,
        "merged": merged,
        "cps": cps,
        "duration": dur,
        "score": (cps > max_cps, cps, abs(dur - 1.8)),
    }


def _choose_merge_candidate(prev_candidate: dict | None, next_candidate: dict | None) -> dict | None:
    candidates = [c for c in (prev_candidate, next_candidate) if c]
    if not candidates:
        return None
    candidates.sort(key=lambda c: c["score"])
    return candidates[0]


def _merge_two_cues(left: dict, right: dict) -> dict:
    merged = {
        "start": min(float(left["start"]), float(right["start"])),
        "end": max(float(left["end"]), float(right["end"])),
        "text": f"{left['text']}\n{right['text']}".strip(),
        "words": list(left.get("words", [])) + list(right.get("words", [])),
    }
    repaired = list(left.get("repaired_zero_duration_texts", [])) + list(right.get("repaired_zero_duration_texts", []))
    if repaired:
        merged["repaired_zero_duration_texts"] = repaired
    return merged


def _cue_duration(cue: dict) -> float:
    return max(0.001, float(cue["end"]) - float(cue["start"]))


def _cue_chars(cue: dict) -> int:
    return len("".join(ch for ch in cue["text"] if ch not in " \n\t"))


def _cue_cps(cue: dict) -> float:
    return _cue_chars(cue) / _cue_duration(cue)


def _split_group(
    words: list[dict], max_cue_s: float, out: list[list[dict]]
):
    dur = words[-1]["end"] - words[0]["start"]
    if dur <= max_cue_s or len(words) < 2:
        out.append(words)
        return

    best_gap = -1.0
    best_idx = -1
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap > best_gap:
            best_gap = gap
            best_idx = i

    if best_gap < 0.01:
        _greedy_split(words, max_cue_s, out)
        return

    left = words[:best_idx]
    right = words[best_idx:]
    _split_group(left, max_cue_s, out)
    _split_group(right, max_cue_s, out)


def _greedy_split(
    words: list[dict], max_cue_s: float, out: list[list[dict]]
):
    current: list[dict] = []
    for w in words:
        if current and w["end"] - current[0]["start"] > max_cue_s:
            out.append(current)
            current = []
        current.append(w)
    if current:
        out.append(current)
