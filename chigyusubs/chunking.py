"""VAD-aware audio chunking for long-form transcription."""


def find_chunk_boundaries(
    rttm_segments: list[dict],
    total_duration: float,
    target_chunk_s: float = 600,
    min_gap_s: float = 2.0,
) -> list[tuple[float, float]]:
    """Find chunk boundaries at natural silence gaps from RTTM/VAD segments.

    Returns list of (start_s, end_s) tuples for each chunk, splitting at
    silence gaps >= min_gap_s that fall nearest to target_chunk_s intervals.
    """
    if not rttm_segments:
        return [(0, total_duration)]

    sorted_segs = sorted(rttm_segments, key=lambda s: s["start"])
    gaps = []
    for i in range(1, len(sorted_segs)):
        gap_start = sorted_segs[i - 1]["end"]
        gap_end = sorted_segs[i]["start"]
        gap_dur = gap_end - gap_start
        if gap_dur >= min_gap_s:
            gaps.append({
                "time": (gap_start + gap_end) / 2,
                "gap_start": gap_start,
                "gap_end": gap_end,
                "duration": gap_dur,
            })

    boundaries: list[tuple[float, float]] = []
    chunk_start = 0.0

    while chunk_start < total_duration:
        target_end = chunk_start + target_chunk_s
        if target_end >= total_duration - target_chunk_s * 0.3:
            boundaries.append((chunk_start, total_duration))
            break

        best_gap = None
        best_dist = float("inf")
        for g in gaps:
            if g["time"] <= chunk_start:
                continue
            dist = abs(g["time"] - target_end)
            if dist < target_chunk_s * 0.4 and dist < best_dist:
                best_dist = dist
                best_gap = g

        if best_gap:
            boundaries.append((chunk_start, best_gap["gap_start"]))
            chunk_start = best_gap["gap_end"]
        else:
            boundaries.append((chunk_start, min(target_end, total_duration)))
            chunk_start = target_end

    return boundaries
