#!/usr/bin/env python3
"""CTC forced alignment using NTQAI/wav2vec2-large-japanese.

Uses torchaudio.functional.forced_align for character-level CTC alignment,
which is more robust than stable-ts cross-attention alignment for short
utterances and chunk edges.

Usage:
  python scripts/align_ctc.py \
    --video samples/episodes/oni_no_dokkiri_de_namida_ep2/source/oni_no_dokkiri_de_namida_ep2.webm \
    --chunks samples/episodes/oni_no_dokkiri_de_namida_ep2/transcription/oni_no_dokkiri_de_namida_ep2_video_only_v2_gemini_raw.json \
    --output-words samples/episodes/oni_no_dokkiri_de_namida_ep2/transcription/oni_no_dokkiri_de_namida_ep2_ctc_words.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "NTQAI/wav2vec2-large-japanese"
_FALLBACK_LINE_SLOT_S = 0.08
CPU_THREADS = 24

# Lines that should be stripped before alignment (visual-only context)
_VISUAL_RE = re.compile(r"^\[画面:.*\]$")
_VISUAL_CAPTURE_RE = re.compile(r"^\[画面:\s*(.*?)\]$")
_SPEAKER_DASH_RE = re.compile(r"^--\s*")
_VISUAL_NARRATION_HINTS = (
    "ご覧の通り",
    "現在地",
    "現在位置",
    "状況",
    "一方",
    "優勢",
    "劣勢",
    "スタートから",
    "ルール",
    "挑戦",
    "設定",
    "被害者",
    "刑事",
    "残り",
    "得点",
    "経過",
    "順位",
    "ここまで",
)


def load_model():
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    print(f"Loading {MODEL_NAME}...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.eval().to(DEVICE)
    print(f"Model on {DEVICE}")
    return model, processor


def configure_torch_threads() -> None:
    """Cap CPU thread fanout for the CPU-side forced alignment step."""
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)


def diagnostics_path(output_path: str | Path) -> Path:
    return Path(f"{output_path}.diagnostics.json")


def extract_audio_slice(video_path, start_s, duration_s, out_path):
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(start_s), "-i", video_path,
            "-t", str(duration_s), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "wav", out_path,
        ],
        capture_output=True, check=True,
    )


def clean_chunk_text(raw_text: str) -> list[dict]:
    """Extract spoken lines from a chunk, preserving Gemini turn boundaries."""
    lines = []
    current_turn = -1
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _VISUAL_RE.match(line):
            continue
        starts_new_turn = bool(_SPEAKER_DASH_RE.match(line))
        text = _SPEAKER_DASH_RE.sub("", line).strip()
        if not text:
            continue
        if starts_new_turn or current_turn < 0:
            current_turn += 1
        lines.append(
            {
                "text": text,
                "turn_index": current_turn,
                "starts_new_turn": starts_new_turn,
            }
        )
    return lines


def inspect_visual_context(raw_text: str) -> dict:
    """Flag chunks where visual-only text may be substituting for narration."""
    items = []
    visual_lines = []
    for raw_line in raw_text.split("\n"):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        visual_match = _VISUAL_CAPTURE_RE.match(raw_line)
        if visual_match:
            text = visual_match.group(1).strip()
            items.append({"type": "visual", "text": text})
            visual_lines.append(text)
            continue
        spoken_text = _SPEAKER_DASH_RE.sub("", raw_line).strip()
        if spoken_text:
            items.append({"type": "spoken", "text": spoken_text})

    suspicious_runs = []
    idx = 0
    while idx < len(items):
        if items[idx]["type"] != "visual":
            idx += 1
            continue
        run_start = idx
        while idx < len(items) and items[idx]["type"] == "visual":
            idx += 1
        run_lines = [item["text"] for item in items[run_start:idx]]
        matched_hints = sorted(
            {
                hint
                for line in run_lines
                for hint in _VISUAL_NARRATION_HINTS
                if hint in line
            }
        )
        has_spoken_before = any(item["type"] == "spoken" for item in items[:run_start])
        has_spoken_after = any(item["type"] == "spoken" for item in items[idx:])
        long_visual_count = sum(1 for line in run_lines if len(line) >= 14)
        if len(run_lines) >= 2 and has_spoken_before and has_spoken_after and (matched_hints or long_visual_count >= 2):
            suspicious_runs.append(
                {
                    "visual_line_count": len(run_lines),
                    "lines": run_lines[:4],
                    "matched_hints": matched_hints,
                    "has_spoken_before": has_spoken_before,
                    "has_spoken_after": has_spoken_after,
                }
            )

    return {
        "visual_line_count": len(visual_lines),
        "visual_lines_sample": visual_lines[:6],
        "narration_like_visual_line_count": sum(
            1
            for line in visual_lines
            if any(hint in line for hint in _VISUAL_NARRATION_HINTS) or len(line) >= 14
        ),
        "possible_visual_narration_substitution": bool(suspicious_runs),
        "suspicious_visual_runs": suspicious_runs[:4],
    }


def align_chunk(model, processor, waveform: torch.Tensor, lines: list[dict]) -> tuple[list[dict], list[dict]]:
    """Align spoken lines to audio using CTC forced alignment.

    Returns list of segment dicts with word-level timestamps.
    """
    if not lines:
        return [], []

    vocab = processor.tokenizer.get_vocab()
    pad_id = processor.tokenizer.pad_token_id

    # Get log probabilities from model
    with torch.no_grad():
        inputs = processor(
            waveform.squeeze().numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(DEVICE)
        logits = model(input_values).logits
        log_probs = torch.log_softmax(logits, dim=-1).cpu()

    num_frames = log_probs.shape[1]
    audio_duration = waveform.shape[-1] / SAMPLE_RATE

    # Build full token sequence from all lines
    full_text = "".join(line["text"] for line in lines)
    token_ids = []
    chars = []
    for ch in full_text:
        tid = vocab.get(ch)
        if tid is not None and tid != pad_id:
            token_ids.append(tid)
            chars.append(ch)

    if not token_ids:
        return [], []

    # Run CTC forced alignment
    targets = torch.tensor([token_ids], dtype=torch.int32)
    input_lengths = torch.tensor([num_frames], dtype=torch.int32)
    target_lengths = torch.tensor([len(token_ids)], dtype=torch.int32)

    try:
        aligned_tokens, scores = torchaudio.functional.forced_align(
            log_probs, targets, input_lengths, target_lengths, blank=pad_id,
        )
    except Exception as e:
        print(f"  forced_align failed: {e}")
        return [], []

    # Merge consecutive aligned frames into token spans
    token_spans = torchaudio.functional.merge_tokens(aligned_tokens[0], scores[0])

    if len(token_spans) != len(chars):
        print(f"  Warning: token_spans ({len(token_spans)}) != characters ({len(chars)}), skipping")
        return [], []

    # Map frame indices to time
    frame_to_time = audio_duration / num_frames

    # Build character-level timestamps
    char_timestamps = []
    for span, char in zip(token_spans, chars):
        t_start = span.start * frame_to_time
        t_end = (span.end + 1) * frame_to_time
        char_timestamps.append({
            "char": char,
            "start": round(t_start, 3),
            "end": round(t_end, 3),
            "score": round(span.score, 4),
        })

    # Reconstruct segments from the original lines with character-level words
    segments = []
    char_idx = 0
    for line_idx, line in enumerate(lines):
        line_text = str(line["text"])
        # First pass: match aligned chars and mark unaligned ones
        raw_entries = []
        for ch in line_text:
            if char_idx < len(char_timestamps) and char_timestamps[char_idx]["char"] == ch:
                raw_entries.append(("aligned", char_timestamps[char_idx]))
                char_idx += 1
            else:
                raw_entries.append(("unaligned", ch))

        # Second pass: attach unaligned chars to nearest aligned char
        # Prefer attaching to previous (e.g. 、。?! after a character)
        line_chars = []
        for kind, val in raw_entries:
            if kind == "aligned":
                line_chars.append({**val})
            else:
                if line_chars:
                    line_chars[-1]["char"] += val
                else:
                    # Leading unaligned — will prepend to next aligned char
                    pass

        # Handle leading unaligned chars by prepending to first aligned char
        leading = []
        for kind, val in raw_entries:
            if kind == "unaligned":
                leading.append(val)
            else:
                break
        if leading and line_chars:
            line_chars[0]["char"] = "".join(leading) + line_chars[0]["char"]

        if not line_chars:
            segments.append({
                "start": 0.0,
                "end": 0.0,
                "text": line_text,
                "_line_index": line_idx,
                "_unaligned": True,
                "turn_index": int(line.get("turn_index", line_idx)),
                "starts_new_turn": bool(line.get("starts_new_turn", False)),
                "words": [{"start": 0.0, "end": 0.0, "word": line_text, "probability": 0.0}],
            })
            continue

        seg_start = line_chars[0]["start"]
        seg_end = line_chars[-1]["end"]

        words = []
        for c in line_chars:
            words.append({
                "start": c["start"],
                "end": c["end"],
                "word": c["char"],
                "probability": c["score"],
            })

        segments.append({
            "start": seg_start,
            "end": seg_end,
            "text": line_text,
            "_line_index": line_idx,
            "turn_index": int(line.get("turn_index", line_idx)),
            "starts_new_turn": bool(line.get("starts_new_turn", False)),
            "words": words,
        })

    repairs = _repair_unaligned_line_timings(segments)
    _enforce_monotonic_timings(segments)
    _finalize_alignment_diagnostics(segments, repairs)

    return segments, repairs


def _repair_unaligned_line_timings(segments: list[dict]) -> list[dict]:
    """Repair all-unaligned lines so they keep local ordering within a chunk.

    When an entire line fails alignment, the old behavior left it at 0.0s.
    After the chunk offset is applied, that becomes the chunk start, which can
    throw a short answer far backward in time and make later reflow drop it.
    Instead, assign a small local slot between neighboring aligned lines.
    """
    if not segments:
        return []

    repairs: list[dict] = []

    def timed(seg: dict) -> bool:
        return float(seg.get("end", 0.0)) > float(seg.get("start", 0.0))

    i = 0
    while i < len(segments):
        if not segments[i].get("_unaligned"):
            i += 1
            continue

        j = i
        while j < len(segments) and segments[j].get("_unaligned"):
            j += 1

        prev_idx = i - 1
        while prev_idx >= 0 and not timed(segments[prev_idx]):
            prev_idx -= 1

        next_idx = j
        while next_idx < len(segments) and not timed(segments[next_idx]):
            next_idx += 1

        cluster = segments[i:j]
        cluster_count = len(cluster)

        if prev_idx >= 0 and next_idx < len(segments):
            gap_start = float(segments[prev_idx]["end"])
            gap_end = float(segments[next_idx]["start"])
            available = max(0.0, gap_end - gap_start)
            if available > 0.0:
                mode = "between_neighbors"
                slot = min(_FALLBACK_LINE_SLOT_S, available / cluster_count)
                slot = max(slot, 0.001)
                cursor = gap_start
                for seg in cluster:
                    original_start = float(seg.get("start", 0.0))
                    original_end = float(seg.get("end", 0.0))
                    seg["start"] = round(cursor, 3)
                    cursor = min(gap_end, cursor + slot)
                    seg["end"] = round(max(cursor, seg["start"] + 0.001), 3)
                    repairs.append({
                        "line_index_in_chunk": int(seg.get("_line_index", -1)),
                        "text": seg.get("text", ""),
                        "repair_mode": mode,
                        "original_local_start_s": round(original_start, 3),
                        "original_local_end_s": round(original_end, 3),
                    })
                i = j
                continue

        if prev_idx >= 0:
            base = float(segments[prev_idx]["end"])
            mode = "after_previous"
        elif next_idx < len(segments):
            base = max(0.0, float(segments[next_idx]["start"]) - cluster_count * _FALLBACK_LINE_SLOT_S)
            mode = "before_next"
        else:
            base = 0.0
            mode = "fallback_from_zero"

        cursor = base
        for seg in cluster:
            original_start = float(seg.get("start", 0.0))
            original_end = float(seg.get("end", 0.0))
            seg["start"] = round(cursor, 3)
            cursor += _FALLBACK_LINE_SLOT_S
            seg["end"] = round(cursor, 3)
            repairs.append({
                "line_index_in_chunk": int(seg.get("_line_index", -1)),
                "text": seg.get("text", ""),
                "repair_mode": mode,
                "original_local_start_s": round(original_start, 3),
                "original_local_end_s": round(original_end, 3),
            })

        i = j

    return repairs


def _finalize_alignment_diagnostics(segments: list[dict], repairs: list[dict]) -> None:
    indexed = {int(seg["_line_index"]): seg for seg in segments if "_line_index" in seg}
    for item in repairs:
        seg = indexed.get(int(item["line_index_in_chunk"]))
        if seg is None:
            continue
        item["repaired_local_start_s"] = round(float(seg["start"]), 3)
        item["repaired_local_end_s"] = round(float(seg["end"]), 3)

    for seg in segments:
        seg.pop("_unaligned", None)
        seg.pop("_line_index", None)


def _enforce_monotonic_timings(segments: list[dict]) -> None:
    """Snap segment and word timings forward to preserve transcript order."""
    prev_end = 0.0
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        orig_dur = max(0.001, end - start)

        if start < prev_end:
            start = prev_end
        if end <= start:
            end = start + orig_dur

        words = seg.get("words") or []
        word_cursor = start
        for word in words:
            w_start = float(word.get("start", 0.0))
            w_end = float(word.get("end", 0.0))
            w_dur = max(0.001, w_end - w_start)

            if w_start < word_cursor:
                w_start = word_cursor
            if w_end <= w_start:
                w_end = w_start + w_dur

            word["start"] = round(w_start, 3)
            word["end"] = round(w_end, 3)
            word_cursor = w_end

        if words:
            start = float(words[0]["start"])
            end = max(end, float(words[-1]["end"]))

        seg["start"] = round(start, 3)
        seg["end"] = round(end, 3)
        prev_end = seg["end"]


def main():
    run = start_run("align_ctc")
    parser = argparse.ArgumentParser(description="CTC forced alignment using wav2vec2 Japanese.")
    parser.add_argument("--video", required=True, help="Input video/audio file.")
    parser.add_argument("--chunks", required=True, help="Gemini raw transcription JSON with chunks.")
    parser.add_argument("--output-words", required=True, help="Output JSON with aligned words.")
    args = parser.parse_args()

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    configure_torch_threads()
    model, processor = load_model()

    all_segments = []
    all_chunk_diagnostics = []

    with tempfile.TemporaryDirectory() as work_dir:
        for i, chunk in enumerate(chunks_data):
            c_start = chunk["chunk_start_s"]
            c_end = chunk["chunk_end_s"]
            c_dur = c_end - c_start
            raw_text = chunk.get("text", "")
            visual_context = inspect_visual_context(raw_text)

            lines = clean_chunk_text(raw_text)
            print(f"\nChunk {i+1}/{len(chunks_data)} ({c_start:.1f}s\u2013{c_end:.1f}s, {len(lines)} lines)")
            if visual_context["possible_visual_narration_substitution"]:
                print(
                    "  advisory: visual-only text may be substituting for narrated speech "
                    f"({len(visual_context['suspicious_visual_runs'])} suspicious run(s))"
                )

            if not lines:
                print("  Skipping empty chunk.")
                all_chunk_diagnostics.append({
                    "chunk": i,
                    "chunk_start_s": c_start,
                    "chunk_end_s": c_end,
                    "line_count": 0,
                    "turn_count": 0,
                    "segments": 0,
                    "words": 0,
                    "zero_duration_segments": 0,
                    "zero_duration_words": 0,
                    "repaired_unaligned_segments": 0,
                    "repaired_unaligned_details": [],
                    "stripped_visual_lines": visual_context["visual_line_count"],
                    "visual_lines_sample": visual_context["visual_lines_sample"],
                    "narration_like_visual_line_count": visual_context["narration_like_visual_line_count"],
                    "possible_visual_narration_substitution": visual_context["possible_visual_narration_substitution"],
                    "suspicious_visual_runs": visual_context["suspicious_visual_runs"],
                    "review_reasons": ["possible_visual_narration_substitution"]
                    if visual_context["possible_visual_narration_substitution"]
                    else [],
                    "needs_review": bool(visual_context["possible_visual_narration_substitution"]),
                    "status": "empty_visual_risk" if visual_context["possible_visual_narration_substitution"] else "empty",
                })
                continue

            # Extract audio slice
            slice_wav = os.path.join(work_dir, f"slice_{i}.wav")
            extract_audio_slice(args.video, c_start, c_dur, slice_wav)

            data, file_sr = sf.read(slice_wav)
            waveform = torch.from_numpy(data).float().unsqueeze(0)
            if file_sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, file_sr, SAMPLE_RATE)

            # Align
            segments, repairs = align_chunk(model, processor, waveform, lines)

            # Offset timestamps to episode time
            for seg in segments:
                seg["start"] = round(seg["start"] + c_start, 3)
                seg["end"] = round(seg["end"] + c_start, 3)
                for w in seg["words"]:
                    w["start"] = round(w["start"] + c_start, 3)
                    w["end"] = round(w["end"] + c_start, 3)
            for item in repairs:
                item["repaired_start_s"] = round(float(item["repaired_local_start_s"]) + c_start, 3)
                item["repaired_end_s"] = round(float(item["repaired_local_end_s"]) + c_start, 3)

            all_segments.extend(segments)

            # Quick stats
            zero_dur = sum(1 for s in segments if s["start"] == s["end"])
            zero_words = sum(1 for s in segments for w in s["words"] if w["start"] == w["end"])
            if repairs:
                print(f"  repaired {len(repairs)}/{len(segments)} all-unaligned segments")
            if zero_dur:
                print(f"  {zero_dur}/{len(segments)} zero-duration segments")
            review_reasons = []
            if repairs:
                review_reasons.append("interpolated_unaligned_segments")
            if zero_dur or zero_words:
                review_reasons.append("zero_duration_alignment")
            if not segments:
                review_reasons.append("no_segments")
            if visual_context["possible_visual_narration_substitution"]:
                review_reasons.append("possible_visual_narration_substitution")
            all_chunk_diagnostics.append({
                "chunk": i,
                "chunk_start_s": c_start,
                "chunk_end_s": c_end,
                "line_count": len(lines),
                "turn_count": len({int(line["turn_index"]) for line in lines}),
                "segments": len(segments),
                "words": sum(len(s["words"]) for s in segments),
                "zero_duration_segments": zero_dur,
                "zero_duration_words": zero_words,
                "repaired_unaligned_segments": len(repairs),
                "repaired_unaligned_details": repairs,
                "stripped_visual_lines": visual_context["visual_line_count"],
                "visual_lines_sample": visual_context["visual_lines_sample"],
                "narration_like_visual_line_count": visual_context["narration_like_visual_line_count"],
                "possible_visual_narration_substitution": visual_context["possible_visual_narration_substitution"],
                "suspicious_visual_runs": visual_context["suspicious_visual_runs"],
                "review_reasons": review_reasons,
                "needs_review": bool(review_reasons),
                "status": (
                    "ok_repaired"
                    if repairs
                    else (
                        "ok_visual_risk"
                        if visual_context["possible_visual_narration_substitution"]
                        else ("ok" if segments else "no_segments")
                    )
                ),
            })

    # Write output
    total_words = sum(len(s["words"]) for s in all_segments)
    zero_segs = sum(1 for s in all_segments if s["start"] == s["end"])
    zero_words = sum(1 for s in all_segments for w in s["words"] if w["start"] == w["end"])
    repaired_segments = sum(item["repaired_unaligned_segments"] for item in all_chunk_diagnostics)
    repaired_chunks = sum(1 for item in all_chunk_diagnostics if item["repaired_unaligned_segments"] > 0)
    visual_risk_chunks = sum(
        1 for item in all_chunk_diagnostics if item.get("possible_visual_narration_substitution")
    )

    print(f"\nResults: {len(all_segments)} segments, {total_words} words")
    print(f"Zero-duration: {zero_segs} segments ({zero_segs/max(len(all_segments),1)*100:.1f}%), "
          f"{zero_words} words ({zero_words/max(total_words,1)*100:.1f}%)")
    print(f"Interpolated all-unaligned segments: {repaired_segments} across {repaired_chunks} chunks")
    print(f"Visual-only narration risk chunks: {visual_risk_chunks}")

    Path(args.output_words).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_words, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)
    print(f"Written to {args.output_words}")
    diag_path = diagnostics_path(args.output_words)
    diag_path.write_text(json.dumps(all_chunk_diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Diagnostics written: {diag_path}")

    metadata = finish_run(
        run,
        inputs={"video": args.video, "chunks_json": args.chunks},
        outputs={"words_json": args.output_words, "alignment_diagnostics_json": str(diag_path)},
        settings={"model": MODEL_NAME, "language": "ja", "cpu_threads": CPU_THREADS},
        stats={
            "chunks_loaded": len(chunks_data),
            "segments_written": len(all_segments),
            "words_written": total_words,
            "source_turns": len({int(seg.get("turn_index", -1)) for seg in all_segments if "turn_index" in seg}),
            "zero_duration_segments": zero_segs,
            "zero_duration_words": zero_words,
            "interpolated_unaligned_segments": repaired_segments,
            "chunks_with_interpolated_unaligned_segments": repaired_chunks,
            "chunks_with_visual_narration_substitution_risk": visual_risk_chunks,
        },
    )
    write_metadata(args.output_words, metadata)
    print(f"Metadata written: {metadata_path(args.output_words)}")


if __name__ == "__main__":
    main()
