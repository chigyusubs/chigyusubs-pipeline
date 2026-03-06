"""Semantic reflow: re-segment word-level timestamps into subtitle cues."""


def reflow_words(
    segments: list[dict],
    pause_threshold: float = 0.3,
    max_cue_s: float = 10.0,
    min_cue_s: float = 0.3,
) -> list[dict]:
    """Re-segment word timestamps into cues aligned to natural pauses.

    Args:
        segments: List of segment dicts with "words" arrays (faster-whisper format).
        pause_threshold: Gap in seconds between words that triggers a cue break.
        max_cue_s: Maximum cue duration; longer groups get split.
        min_cue_s: Minimum cue duration; shorter groups merge into the next.

    Returns:
        List of cue dicts with keys: start, end, text, words.
    """
    all_words = []
    for seg in segments:
        for w in seg.get("words", []):
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

    return cues


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
