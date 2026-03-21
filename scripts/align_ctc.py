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

from chigyusubs.metadata import (
    finish_run,
    inherit_run_id,
    lineage_output_path,
    metadata_path,
    start_run,
    update_preferred_manifest,
    write_metadata,
)

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "NTQAI/wav2vec2-large-japanese"
RESCUE_MODEL_NAME = "large-v3"
_FALLBACK_LINE_SLOT_S = 0.08
_WEAK_ANCHOR_VOCAB_THRESHOLD = 0.4
_RESCUE_CONTEXT_NEIGHBORS = 2
_RESCUE_CLIP_PAD_S = 1.0
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

    # Build full token sequence from all lines, tracking per-line vocab coverage
    full_text = "".join(line["text"] for line in lines)
    token_ids = []
    chars = []
    for ch in full_text:
        tid = vocab.get(ch)
        if tid is not None and tid != pad_id:
            token_ids.append(tid)
            chars.append(ch)

    # Per-line vocab coverage ratio (fraction of chars the CTC model can tokenize)
    for line in lines:
        line_chars = len(line["text"])
        line_hits = sum(1 for ch in line["text"] if vocab.get(ch) is not None and vocab.get(ch) != pad_id)
        line["_vocab_ratio"] = round(line_hits / max(line_chars, 1), 3)

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
                "_vocab_ratio": float(line.get("_vocab_ratio", 0.0)),
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
            "_vocab_ratio": float(line.get("_vocab_ratio", 1.0)),
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


def _is_weak_anchor(seg: dict) -> bool:
    """Check if a segment is a weak anchor based on vocab coverage and duration."""
    vocab_ratio = float(seg.get("_vocab_ratio", 1.0))
    if vocab_ratio < _WEAK_ANCHOR_VOCAB_THRESHOLD:
        return True
    # Zero-duration with very short text is also suspect
    dur = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
    if dur <= 0.0 and len(seg.get("text", "")) <= 10:
        return True
    return False


def _build_rescue_groups(
    segments: list[dict],
    weak_indices: list[int],
) -> list[dict]:
    """Cluster nearby weak segments and attach strong neighbors as context."""
    if not weak_indices:
        return []

    # Cluster weak indices that are consecutive or within 2 positions
    clusters: list[list[int]] = []
    current = [weak_indices[0]]
    for idx in weak_indices[1:]:
        if idx - current[-1] <= _RESCUE_CONTEXT_NEIGHBORS:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)

    groups = []
    for cluster in clusters:
        first_weak = cluster[0]
        last_weak = cluster[-1]

        # Grab strong neighbors on each side for context
        ctx_start = max(0, first_weak - _RESCUE_CONTEXT_NEIGHBORS)
        ctx_end = min(len(segments) - 1, last_weak + _RESCUE_CONTEXT_NEIGHBORS)

        # Find valid clip boundaries — expand outward if context neighbors
        # are also zero-duration (they might be weak too)
        clip_left = ctx_start
        while clip_left > 0 and segments[clip_left]["end"] <= segments[clip_left]["start"]:
            clip_left -= 1
        clip_right = ctx_end
        while clip_right < len(segments) - 1 and segments[clip_right]["end"] <= segments[clip_right]["start"]:
            clip_right += 1

        clip_start = float(segments[clip_left]["start"]) - _RESCUE_CLIP_PAD_S
        clip_end = float(segments[clip_right]["end"]) + _RESCUE_CLIP_PAD_S
        clip_start = max(0.0, clip_start)

        groups.append({
            "ctx_range": (ctx_start, ctx_end),
            "weak_indices": set(cluster),
            "clip_start": clip_start,
            "clip_end": clip_end,
        })
    return groups


def _run_rescue_pass(
    segments: list[dict],
    video_path: str,
    rescue_model_name: str = RESCUE_MODEL_NAME,
) -> list[dict]:
    """Run stable-ts alignment rescue on weak-anchor segments.

    Returns a list of rescue detail dicts for diagnostics.
    """
    weak_indices = [i for i, seg in enumerate(segments) if _is_weak_anchor(seg)]
    if not weak_indices:
        print("\nRescue pass: no weak-anchor segments detected.")
        return []

    print(f"\nRescue pass: {len(weak_indices)} weak-anchor segments detected")
    for i in weak_indices:
        seg = segments[i]
        print(f"  [{i}] vocab={seg.get('_vocab_ratio', '?'):.2f}  "
              f"dur={float(seg['end']) - float(seg['start']):.3f}s  "
              f"{seg['text']!r}")

    groups = _build_rescue_groups(segments, weak_indices)
    print(f"  Built {len(groups)} rescue group(s)")

    import stable_whisper

    print(f"  Loading stable-ts model: {rescue_model_name}")
    rescue_model = stable_whisper.load_model(rescue_model_name, device=DEVICE)

    rescue_details: list[dict] = []

    with tempfile.TemporaryDirectory() as work_dir:
        for gi, group in enumerate(groups):
            ctx_start, ctx_end = group["ctx_range"]
            group_segs = segments[ctx_start:ctx_end + 1]

            # Build text for the full context group
            group_text = "\n".join(seg["text"] for seg in group_segs)
            clip_start = group["clip_start"]
            clip_end = group["clip_end"]

            print(f"\n  Rescue group {gi + 1}/{len(groups)}: "
                  f"segs [{ctx_start}..{ctx_end}], "
                  f"clip {clip_start:.1f}-{clip_end:.1f}s")

            # Extract audio clip
            clip_path = os.path.join(work_dir, f"rescue_{gi}.wav")
            try:
                extract_audio_slice(video_path, clip_start, clip_end - clip_start, clip_path)
            except subprocess.CalledProcessError:
                print(f"    ffmpeg clip extraction failed, skipping group")
                continue

            # Run stable-ts alignment
            try:
                result = rescue_model.align(
                    clip_path,
                    group_text,
                    language="ja",
                )
            except Exception as e:
                print(f"    stable-ts align failed: {e}")
                continue

            if result is None:
                print(f"    stable-ts returned None")
                continue

            # Collect all words from result, offset to episode time
            rescue_words = []
            for rs in result.segments:
                for w in rs.words:
                    rescue_words.append({
                        "word": w.word,
                        "start": round(w.start + clip_start, 3),
                        "end": round(w.end + clip_start, 3),
                        "probability": round(getattr(w, "probability", 0.0), 4),
                    })

            # Map rescue words back to each segment by walking through text
            _apply_rescue_to_group(
                segments, group, rescue_words, rescue_details,
            )

    del rescue_model
    torch.cuda.empty_cache()

    rescued = sum(1 for d in rescue_details if d["status"] == "rescued")
    failed = sum(1 for d in rescue_details if d["status"] == "rescue_failed")
    print(f"\nRescue pass complete: {rescued} rescued, {failed} failed")
    return rescue_details


def _apply_rescue_to_group(
    segments: list[dict],
    group: dict,
    rescue_words: list[dict],
    rescue_details: list[dict],
):
    """Map stable-ts rescue words back to segments, updating only weak ones."""
    ctx_start, ctx_end = group["ctx_range"]
    weak_indices = group["weak_indices"]

    # Walk through rescue words and match to each segment's text in order.
    # We consume rescue words character by character, matching against each
    # segment's text to determine which words belong to which segment.
    word_idx = 0
    rescue_text_flat = "".join(w["word"] for w in rescue_words)

    for seg_idx in range(ctx_start, ctx_end + 1):
        seg = segments[seg_idx]
        seg_text = seg["text"]

        # Find which rescue words cover this segment's text
        seg_word_start = word_idx
        chars_remaining = len(seg_text.replace("\n", ""))
        chars_consumed = 0

        while word_idx < len(rescue_words) and chars_consumed < chars_remaining:
            w_text = rescue_words[word_idx]["word"].replace("\n", "")
            chars_consumed += len(w_text)
            word_idx += 1

        seg_rescue_words = rescue_words[seg_word_start:word_idx]

        if seg_idx not in weak_indices:
            continue

        # Filter to words with non-zero duration
        timed_words = [w for w in seg_rescue_words if w["end"] > w["start"]]

        original_start = float(seg["start"])
        original_end = float(seg["end"])
        original_dur = original_end - original_start

        detail = {
            "segment_index": seg_idx,
            "text": seg_text,
            "vocab_ratio": float(seg.get("_vocab_ratio", 0.0)),
            "original_start": original_start,
            "original_end": original_end,
            "original_duration": round(original_dur, 3),
            "rescue_words": seg_rescue_words,
        }

        if not timed_words:
            detail["status"] = "rescue_failed"
            detail["reason"] = "no timed words from stable-ts"
            rescue_details.append(detail)
            print(f"    [{seg_idx}] FAILED: {seg_text!r} — no timed rescue words")
            continue

        new_start = timed_words[0]["start"]
        new_end = timed_words[-1]["end"]
        new_dur = new_end - new_start

        # Sanity: rescue result should have positive duration
        if new_dur <= 0.0:
            detail["status"] = "rescue_failed"
            detail["reason"] = "rescue duration <= 0"
            rescue_details.append(detail)
            print(f"    [{seg_idx}] FAILED: {seg_text!r} — zero rescue duration")
            continue

        # Apply rescued timestamps
        seg["start"] = round(new_start, 3)
        seg["end"] = round(new_end, 3)

        # Replace word-level timestamps with rescue words
        seg["words"] = [{
            "start": w["start"],
            "end": w["end"],
            "word": w["word"],
            "probability": w["probability"],
        } for w in seg_rescue_words]

        seg["_rescue"] = {
            "method": "stable_ts",
            "model": RESCUE_MODEL_NAME,
            "original_start": original_start,
            "original_end": original_end,
        }

        detail["status"] = "rescued"
        detail["new_start"] = round(new_start, 3)
        detail["new_end"] = round(new_end, 3)
        detail["new_duration"] = round(new_dur, 3)
        rescue_details.append(detail)
        print(f"    [{seg_idx}] RESCUED: {seg_text!r}  "
              f"{original_start:.3f}-{original_end:.3f} → "
              f"{new_start:.3f}-{new_end:.3f}")


def main():
    run = start_run("align_ctc")
    parser = argparse.ArgumentParser(description="CTC forced alignment using wav2vec2 Japanese.")
    parser.add_argument("--video", required=True, help="Input video/audio file.")
    parser.add_argument("--chunks", required=True, help="Gemini raw transcription JSON with chunks.")
    parser.add_argument("--output-words", required=True, help="Output JSON with aligned words.")
    parser.add_argument(
        "--fresh-run-id",
        action="store_true",
        help="Start a new lineage root for this alignment run while preserving the chunk JSON as lineage_source.",
    )
    parser.add_argument("--no-rescue", action="store_true", help="Skip stable-ts rescue pass for weak-anchor segments.")
    args = parser.parse_args()
    if args.fresh_run_id:
        run["lineage_source"] = str(args.chunks)
    else:
        run = inherit_run_id(run, args.chunks)
    requested_output = Path(args.output_words)
    if requested_output.parent.name == "transcription" and not requested_output.name.startswith(f"{run['run_id']}_"):
        args.output_words = str(
            lineage_output_path(
                requested_output.parent,
                artifact_type="ctc_words",
                run=run,
                suffix=requested_output.suffix or ".json",
            )
        )

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

    # Rescue weak-anchor segments with stable-ts
    rescue_details: list[dict] = []
    if not args.no_rescue:
        # Unload CTC model to free VRAM for stable-ts
        del model, processor
        torch.cuda.empty_cache()

        rescue_details = _run_rescue_pass(all_segments, args.video)
    else:
        weak_count = sum(1 for seg in all_segments if _is_weak_anchor(seg))
        if weak_count:
            print(f"\nRescue pass skipped (--no-rescue): {weak_count} weak-anchor segments")

    # Rescue can move weak-anchor segments backward past neighboring transcript
    # lines, so re-assert monotonic order before serializing artifacts.
    _enforce_monotonic_timings(all_segments)

    # Strip internal fields before writing
    for seg in all_segments:
        rescue_info = seg.pop("_rescue", None)
        vocab_ratio = seg.pop("_vocab_ratio", None)
        if rescue_info:
            seg["rescue"] = rescue_info
        if vocab_ratio is not None and vocab_ratio < _WEAK_ANCHOR_VOCAB_THRESHOLD:
            seg["anchor_confidence"] = "weak"
            seg["vocab_ratio"] = vocab_ratio

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
    rescued_count = sum(1 for d in rescue_details if d["status"] == "rescued")
    rescue_failed_count = sum(1 for d in rescue_details if d["status"] == "rescue_failed")
    weak_anchor_count = sum(1 for seg in all_segments if seg.get("anchor_confidence") == "weak")
    if rescue_details:
        print(f"Weak-anchor rescue: {rescued_count} rescued, {rescue_failed_count} failed")

    Path(args.output_words).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_words, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)
    print(f"Written to {args.output_words}")
    diag_path = diagnostics_path(args.output_words)
    full_diagnostics = {
        "chunks": all_chunk_diagnostics,
    }
    if rescue_details:
        full_diagnostics["rescue"] = {
            "model": RESCUE_MODEL_NAME,
            "weak_anchor_threshold": _WEAK_ANCHOR_VOCAB_THRESHOLD,
            "segments_attempted": len(rescue_details),
            "segments_rescued": rescued_count,
            "segments_failed": rescue_failed_count,
            "details": rescue_details,
        }
    diag_path.write_text(json.dumps(full_diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Diagnostics written: {diag_path}")

    metadata = finish_run(
        run,
        inputs={"video": args.video, "chunks_json": args.chunks},
        outputs={"words_json": args.output_words, "alignment_diagnostics_json": str(diag_path)},
        settings={
            "model": MODEL_NAME,
            "language": "ja",
            "cpu_threads": CPU_THREADS,
            "rescue_model": None if args.no_rescue else RESCUE_MODEL_NAME,
            "weak_anchor_threshold": _WEAK_ANCHOR_VOCAB_THRESHOLD,
        },
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
            "weak_anchor_segments": weak_anchor_count,
            "weak_anchor_rescued": rescued_count,
            "weak_anchor_rescue_failed": rescue_failed_count,
        },
    )
    write_metadata(args.output_words, metadata)
    if Path(args.output_words).parent.name == "transcription":
        update_preferred_manifest(Path(args.output_words).parent, ctc_words=Path(args.output_words).name)
    print(f"Metadata written: {metadata_path(args.output_words)}")


if __name__ == "__main__":
    main()
