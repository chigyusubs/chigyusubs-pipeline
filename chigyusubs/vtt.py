"""VTT and JSON output formatting utilities."""

import json


def format_ts(seconds: float) -> str:
    """Format seconds as VTT timestamp HH:MM:SS.mmm or MM:SS.mmm."""
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{int(hours):02d}:{int(mins):02d}:{secs:06.3f}"
    return f"{int(mins):02d}:{secs:06.3f}"


def format_ts_full(seconds: float) -> str:
    """Format seconds as VTT timestamp HH:MM:SS.mmm (always with hours)."""
    total_ms = round(seconds * 1000)
    h = total_ms // 3600000
    mi = (total_ms % 3600000) // 60000
    s = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{mi:02d}:{s:02d}.{ms:03d}"


def write_vtt(cues: list[dict], output_path: str, include_speaker: bool = False):
    """Write cues as a standard VTT file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for cue in cues:
            start = format_ts(cue["start"])
            end = format_ts(cue["end"])
            f.write(f"{start} --> {end}\n")
            speaker = cue.get("speaker", "") if include_speaker else ""
            text = cue["text"]
            if speaker:
                f.write(f"{speaker}: {text}\n\n")
            else:
                f.write(f"{text}\n\n")


def _seg_get(seg, key):
    """Access a segment field whether it's a dict or an object."""
    return seg[key] if isinstance(seg, dict) else getattr(seg, key)


def write_standard_vtt(segments, output_path: str):
    """Write a standard VTT file from segment dicts or faster-whisper objects."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = format_ts(_seg_get(seg, "start"))
            end = format_ts(_seg_get(seg, "end"))
            text = _seg_get(seg, "text").strip()
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")


def write_word_timestamps_json(segments, output_path: str):
    """Write detailed word timestamps to a JSON file."""
    data = []
    for seg in segments:
        words_raw = _seg_get(seg, "words") or []
        words_out = []
        for w in words_raw:
            if isinstance(w, dict):
                words_out.append(w)
            else:
                words_out.append({
                    "start": w.start, "end": w.end,
                    "word": w.word, "probability": w.probability,
                })
        data.append({
            "start": _seg_get(seg, "start"),
            "end": _seg_get(seg, "end"),
            "text": _seg_get(seg, "text"),
            "words": words_out,
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
