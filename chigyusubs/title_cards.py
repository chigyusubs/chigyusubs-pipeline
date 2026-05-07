"""Candidate generation for title-card augmentation of translated VTT.

Pulls together TransNetV2 shot brackets and Flash Lite chunk OCR to find
shots that look like full-screen title cards. The chunk OCR is coarse
(`timing_basis: chunk_span`, ~240 s windows) so we use it as a *menu* of
plausible card texts per chunk; Codex picks one (or skips) at decision time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from chigyusubs.translation import Cue


CARD_KIND_WHITELIST = {"title_card", "name_card", "info_card"}
CARD_IMPORTANCE_WHITELIST = {"high", "medium"}


@dataclass
class OcrChunk:
    chunk_index: int
    start_s: float
    end_s: float
    items: list[dict]


@dataclass
class Candidate:
    id: int
    shot_start: float
    shot_end: float
    shot_duration: float
    dialog_overlap: float
    ocr_menu: list[dict]
    ja_context_before: list[dict] = field(default_factory=list)
    ja_context_after: list[dict] = field(default_factory=list)
    en_context_before: list[dict] = field(default_factory=list)
    en_context_after: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "shot_start": round(self.shot_start, 3),
            "shot_end": round(self.shot_end, 3),
            "shot_duration": round(self.shot_duration, 3),
            "dialog_overlap": round(self.dialog_overlap, 3),
            "ocr_menu": self.ocr_menu,
            "ja_context_before": self.ja_context_before,
            "ja_context_after": self.ja_context_after,
            "en_context_before": self.en_context_before,
            "en_context_after": self.en_context_after,
        }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_chunk_ocr(path: str | Path) -> list[OcrChunk]:
    """Parse `*_flash_lite_chunk_ocr.json` into a flat list of chunks."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    chunks: list[OcrChunk] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        items = [it for it in entry.get("items", []) if isinstance(it, dict)]
        chunks.append(
            OcrChunk(
                chunk_index=int(entry.get("chunk", len(chunks))),
                start_s=float(entry.get("chunk_start_s", 0.0)),
                end_s=float(entry.get("chunk_end_s", 0.0)),
                items=items,
            )
        )
    return chunks


def load_shots(path: str | Path) -> list[dict]:
    """Parse `*.shots.json` into a list of shot dicts."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("shots", [])
    return [dict(s) for s in raw]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _dialog_overlap_fraction(shot_start: float, shot_end: float, cues: list[Cue]) -> float:
    duration = max(0.001, shot_end - shot_start)
    covered = 0.0
    for cue in cues:
        if cue.end <= shot_start:
            continue
        if cue.start >= shot_end:
            break
        covered += _interval_overlap(shot_start, shot_end, cue.start, cue.end)
    return min(1.0, covered / duration)


def _enclosing_chunk(t: float, chunks: list[OcrChunk]) -> OcrChunk | None:
    for c in chunks:
        if c.start_s <= t < c.end_s:
            return c
    # Edge: the very last chunk's end_s may be exclusive, accept t == end_s.
    if chunks and abs(t - chunks[-1].end_s) < 1e-3:
        return chunks[-1]
    return None


def _menu_for_chunk(chunk: OcrChunk) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for it in chunk.items:
        kind = str(it.get("kind_guess", ""))
        importance = str(it.get("importance", ""))
        text = str(it.get("text", "")).strip()
        if not text:
            continue
        if kind not in CARD_KIND_WHITELIST:
            continue
        if importance not in CARD_IMPORTANCE_WHITELIST:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(
            {
                "text": text,
                "kind_guess": kind,
                "importance": importance,
            }
        )
    return out


def _context_cues(
    cues: list[Cue],
    *,
    before_t: float,
    after_t: float,
    n: int,
) -> tuple[list[dict], list[dict]]:
    before: list[dict] = []
    after: list[dict] = []
    for cue in cues:
        if cue.end <= before_t:
            before.append(_cue_dict(cue))
        elif cue.start >= after_t:
            after.append(_cue_dict(cue))
            if len(after) >= n:
                break
    before = before[-n:]
    return before, after


def _cue_dict(cue: Cue) -> dict:
    return {
        "start": round(cue.start, 3),
        "end": round(cue.end, 3),
        "text": cue.text,
    }


# ---------------------------------------------------------------------------
# Candidate generator
# ---------------------------------------------------------------------------


def find_candidates(
    shots: list[dict],
    chunk_ocr: list[OcrChunk],
    reflow_cues: list[Cue],
    en_cues: list[Cue],
    *,
    min_dur: float = 0.6,
    max_dur: float = 8.0,
    dialog_overlap_max: float = 0.25,
    context_n: int = 2,
) -> list[Candidate]:
    """Find shots that look card-shaped and pair each with the chunk's OCR menu."""
    reflow_sorted = sorted(reflow_cues, key=lambda c: c.start)
    en_sorted = sorted(en_cues, key=lambda c: c.start)

    candidates: list[Candidate] = []
    next_id = 0
    for shot in shots:
        start = float(shot.get("start_time", 0.0))
        end = float(shot.get("end_time", 0.0))
        duration = end - start
        if duration < min_dur or duration > max_dur:
            continue

        overlap = _dialog_overlap_fraction(start, end, reflow_sorted)
        if overlap > dialog_overlap_max:
            continue

        chunk = _enclosing_chunk(start, chunk_ocr)
        if chunk is None:
            continue
        menu = _menu_for_chunk(chunk)
        if not menu:
            continue

        ja_before, ja_after = _context_cues(
            reflow_sorted, before_t=start, after_t=end, n=context_n
        )
        en_before, en_after = _context_cues(
            en_sorted, before_t=start, after_t=end, n=context_n
        )

        candidates.append(
            Candidate(
                id=next_id,
                shot_start=start,
                shot_end=end,
                shot_duration=duration,
                dialog_overlap=overlap,
                ocr_menu=menu,
                ja_context_before=ja_before,
                ja_context_after=ja_after,
                en_context_before=en_before,
                en_context_after=en_after,
            )
        )
        next_id += 1

    return candidates


# ---------------------------------------------------------------------------
# Splicing
# ---------------------------------------------------------------------------


def inject_cards(
    en_cues: list[Cue],
    decisions: Iterable[dict],
    *,
    min_card_duration: float = 0.6,
    margin: float = 0.04,
) -> tuple[list[Cue], list[dict]]:
    """Splice approved card cues into the EN VTT.

    Each decision (action == "include") needs `shot_start`, `shot_end`, `text`.
    Card cues are clamped to the gap around them so they never overlap an
    existing EN cue. If the clamped duration is < `min_card_duration`, the
    card is dropped and the reason recorded.

    Returns (new_cues, audit_entries).
    """
    base = sorted(en_cues, key=lambda c: c.start)
    audit: list[dict] = []

    new_cards: list[Cue] = []
    cards = [d for d in decisions if d.get("action") == "include" and d.get("text")]
    cards.sort(key=lambda d: float(d.get("shot_start", 0.0)))

    for d in cards:
        shot_start = float(d["shot_start"])
        shot_end = float(d["shot_end"])
        text = str(d["text"]).strip()

        # Clamp against existing EN cues
        bound_start, bound_end = _free_window(base + new_cards, shot_start, shot_end, margin)
        if bound_end - bound_start < min_card_duration:
            audit.append(
                {
                    "candidate_id": d.get("candidate_id"),
                    "action": "dropped_no_room",
                    "shot_start": shot_start,
                    "shot_end": shot_end,
                    "free_window": [bound_start, bound_end],
                }
            )
            continue

        cue = Cue(start=bound_start, end=bound_end, text=text)
        new_cards.append(cue)
        audit.append(
            {
                "candidate_id": d.get("candidate_id"),
                "action": "included",
                "start": bound_start,
                "end": bound_end,
                "text": text,
            }
        )

    merged = sorted(base + new_cards, key=lambda c: c.start)
    return merged, audit


def _free_window(
    cues: list[Cue],
    shot_start: float,
    shot_end: float,
    margin: float,
) -> tuple[float, float]:
    """Shrink [shot_start, shot_end] so it doesn't overlap any cue in `cues`."""
    bound_start = shot_start
    bound_end = shot_end
    for cue in cues:
        if cue.end <= bound_start:
            continue
        if cue.start >= bound_end:
            break
        # Overlap. Pick the side with more room.
        left_room = cue.start - bound_start
        right_room = bound_end - cue.end
        if left_room >= right_room:
            bound_end = max(bound_start, cue.start - margin)
        else:
            bound_start = min(bound_end, cue.end + margin)
        if bound_end - bound_start <= 0:
            break
    return bound_start, bound_end
