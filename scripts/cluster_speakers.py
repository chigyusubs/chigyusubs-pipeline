#!/usr/bin/env python3
"""Cluster speaker turns by voice embedding using SpeechBrain ECAPA-TDNN.

Takes CTC-aligned words JSON (with turn_index per segment) and the source
video/audio, extracts per-turn audio clips, computes speaker embeddings,
and clusters them. Outputs a speaker map JSON artifact for downstream
translation disambiguation.

Usage:
  python3.12 scripts/cluster_speakers.py \
    --words samples/episodes/<slug>/transcription/<run>_ctc_words.json \
    --video samples/episodes/<slug>/source/video.mp4 \
    --output samples/episodes/<slug>/transcription/<run>_speaker_map.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Compat shim: torchaudio nightly (2.11+) dropped list_audio_backends
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from speechbrain.inference.speaker import EncoderClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
_MIN_EMBED_DURATION_S = 0.5  # Minimum to extract an embedding at all
_ANCHOR_DURATION_S = 2.0  # Minimum for high-confidence anchor turns
_EMBEDDING_DIM = 192


def load_turns(words_path: str) -> list[dict]:
    """Group segments by speaker turn into turn-level entries.

    turn_index in the CTC words JSON resets at each Gemini chunk boundary,
    so we detect resets and assign globally unique turn IDs.
    """
    with open(words_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Assign globally unique turn IDs by detecting turn_index resets
    global_turn = 0
    prev_turn_idx = -1
    turns_by_global: dict[int, list[dict]] = defaultdict(list)

    for seg in segments:
        turn_idx = int(seg.get("turn_index", -1))
        # New global turn on: reset, new turn_index, or explicit turn start
        if turn_idx < prev_turn_idx or turn_idx != prev_turn_idx:
            if prev_turn_idx >= 0:
                global_turn += 1
        prev_turn_idx = turn_idx
        turns_by_global[global_turn].append(seg)

    turns = []
    for g_turn in sorted(turns_by_global):
        segs = turns_by_global[g_turn]
        start = min(float(s["start"]) for s in segs)
        end = max(float(s["end"]) for s in segs)
        text = " ".join(s.get("text", "") for s in segs).strip()
        turns.append({
            "turn_index": g_turn,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "text": text,
            "segment_count": len(segs),
        })
    return turns


def extract_full_audio(video_path: str, out_path: str):
    """Extract full audio as 16kHz mono WAV."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "wav", out_path,
        ],
        capture_output=True, check=True,
    )


def extract_turn_waveforms(
    audio_path: str,
    turns: list[dict],
    min_duration_s: float = _MIN_EMBED_DURATION_S,
) -> tuple[list[torch.Tensor], list[int]]:
    """Load full audio and extract per-turn waveform slices.

    Returns (waveforms, turn_indices) for turns meeting the minimum duration.
    """
    data, sr = sf.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(data).unsqueeze(0)  # (1, samples)
    if waveform.ndim == 3:
        waveform = waveform.squeeze(-1)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    total_samples = waveform.shape[1]
    waveforms = []
    indices = []

    for turn in turns:
        if turn["duration"] < min_duration_s:
            continue
        start_sample = int(turn["start"] * SAMPLE_RATE)
        end_sample = int(turn["end"] * SAMPLE_RATE)
        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(start_sample, min(end_sample, total_samples))
        if end_sample - start_sample < int(min_duration_s * SAMPLE_RATE):
            continue
        clip = waveform[:, start_sample:end_sample]
        waveforms.append(clip)
        indices.append(turn["turn_index"])

    return waveforms, indices


def compute_embeddings(
    model: EncoderClassifier,
    waveforms: list[torch.Tensor],
    max_samples: int = 16000 * 10,
) -> np.ndarray:
    """Compute speaker embeddings for a list of waveform clips.

    Processes one turn at a time to avoid OOM from padding variable-length
    turns into a single batch. Clips longer than max_samples are truncated
    to the center region (speaker identity is time-invariant).
    """
    all_embeddings = []

    for i, wav in enumerate(waveforms):
        # Truncate to center region if too long
        n_samples = wav.shape[1]
        if n_samples > max_samples:
            offset = (n_samples - max_samples) // 2
            wav = wav[:, offset:offset + max_samples]

        padded = wav.squeeze(0).unsqueeze(0).to(DEVICE)  # (1, samples)
        lengths = torch.ones(1, device=DEVICE)

        with torch.no_grad():
            emb = model.encode_batch(padded, lengths)
            emb = emb.squeeze(0).squeeze(0).cpu().numpy()  # (embedding_dim,)
            all_embeddings.append(emb)

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(waveforms)} turns embedded")

    return np.stack(all_embeddings)


def cluster_speakers_two_pass(
    embeddings: np.ndarray,
    turn_durations: np.ndarray,
    anchor_duration_s: float = _ANCHOR_DURATION_S,
    threshold: float = 0.7,
    min_cluster_size: int = 3,
    assign_threshold: float = 0.75,
) -> np.ndarray:
    """Two-pass speaker clustering.

    Pass 1: Cluster only anchor turns (>= anchor_duration_s) where embeddings
    are reliable. Uses agglomerative clustering with cosine distance.

    Pass 2: Assign shorter turns to their nearest anchor cluster centroid,
    if the cosine distance is below assign_threshold. Otherwise mark as -1.

    Args:
        embeddings: (N, D) array of speaker embeddings.
        turn_durations: (N,) array of turn durations in seconds.
        anchor_duration_s: Minimum duration for anchor turns.
        threshold: Distance threshold for anchor clustering.
        min_cluster_size: Minimum anchor turns to form a cluster.
        assign_threshold: Max cosine distance for assigning short turns.

    Returns:
        Array of cluster labels (N,). Unassigned turns get label -1.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_distances

    n = len(embeddings)
    labels = np.full(n, -1, dtype=int)

    # Pass 1: identify and cluster anchor turns
    anchor_mask = turn_durations >= anchor_duration_s
    anchor_indices = np.where(anchor_mask)[0]
    print(f"  Pass 1: {len(anchor_indices)} anchor turns (>= {anchor_duration_s}s)")

    if len(anchor_indices) < 2:
        print("  Not enough anchor turns to cluster")
        return labels

    anchor_embeddings = embeddings[anchor_indices]
    anchor_dists = cosine_distances(anchor_embeddings)

    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=threshold,
        n_clusters=None,
    )
    anchor_labels = clustering.fit_predict(anchor_dists)

    # Drop small clusters
    counts = np.bincount(anchor_labels[anchor_labels >= 0])
    for label_id, count in enumerate(counts):
        if count < min_cluster_size:
            anchor_labels[anchor_labels == label_id] = -1

    # Compact labels by descending cluster size
    unique_labels = sorted(set(anchor_labels[anchor_labels >= 0]))
    sizes = [(lbl, np.sum(anchor_labels == lbl)) for lbl in unique_labels]
    sizes.sort(key=lambda x: -x[1])
    remap = {old: new for new, (old, _) in enumerate(sizes)}
    remap[-1] = -1
    anchor_labels = np.array([remap[lbl] for lbl in anchor_labels])

    # Apply anchor labels
    for i, global_idx in enumerate(anchor_indices):
        labels[global_idx] = anchor_labels[i]

    n_anchor_clusters = len(set(anchor_labels[anchor_labels >= 0]))
    n_anchor_assigned = int(np.sum(anchor_labels >= 0))
    print(f"  Pass 1: {n_anchor_clusters} clusters, "
          f"{n_anchor_assigned}/{len(anchor_indices)} anchors assigned")

    if n_anchor_clusters == 0:
        return labels

    # Compute cluster centroids from assigned anchors
    centroids = {}
    for cluster_id in range(n_anchor_clusters):
        cluster_mask = anchor_labels == cluster_id
        cluster_embs = anchor_embeddings[cluster_mask]
        centroids[cluster_id] = cluster_embs.mean(axis=0)

    centroid_matrix = np.stack([centroids[i] for i in range(n_anchor_clusters)])

    # Pass 2: assign non-anchor turns to nearest centroid
    non_anchor_indices = np.where(~anchor_mask)[0]
    # Also include unassigned anchors
    unassigned_anchor_indices = anchor_indices[anchor_labels < 0]
    assign_indices = np.concatenate([non_anchor_indices, unassigned_anchor_indices])

    if len(assign_indices) > 0:
        assign_embeddings = embeddings[assign_indices]
        dists_to_centroids = cosine_distances(assign_embeddings, centroid_matrix)

        assigned_count = 0
        for i, global_idx in enumerate(assign_indices):
            nearest = int(np.argmin(dists_to_centroids[i]))
            dist = dists_to_centroids[i, nearest]
            if dist <= assign_threshold:
                labels[global_idx] = nearest
                assigned_count += 1

        print(f"  Pass 2: {assigned_count}/{len(assign_indices)} short/unassigned turns assigned "
              f"(threshold={assign_threshold})")

    return labels


def build_speaker_map(
    turns: list[dict],
    embedded_indices: list[int],
    labels: np.ndarray,
    embeddings: np.ndarray,
) -> dict:
    """Build the speaker map artifact."""
    # Map turn_index -> (label, embedding)
    turn_labels = {}
    turn_embeddings = {}
    for i, turn_idx in enumerate(embedded_indices):
        turn_labels[turn_idx] = int(labels[i])
        turn_embeddings[turn_idx] = embeddings[i]

    # Compute per-speaker stats
    speaker_stats: dict[int, dict] = defaultdict(lambda: {
        "turn_count": 0,
        "total_duration_s": 0.0,
        "segment_count": 0,
    })

    turn_entries = []
    for turn in turns:
        tidx = turn["turn_index"]
        label = turn_labels.get(tidx)
        speaker = f"spk_{label}" if label is not None and label >= 0 else None

        entry = {
            "turn_index": tidx,
            "start": turn["start"],
            "end": turn["end"],
            "duration": turn["duration"],
            "speaker": speaker,
            "text": turn["text"],
        }
        turn_entries.append(entry)

        if speaker is not None:
            stats = speaker_stats[label]
            stats["turn_count"] += 1
            stats["total_duration_s"] += turn["duration"]
            stats["segment_count"] += turn["segment_count"]

    speakers = {}
    for label in sorted(speaker_stats):
        stats = speaker_stats[label]
        stats["total_duration_s"] = round(stats["total_duration_s"], 1)
        speakers[f"spk_{label}"] = stats

    assigned = sum(1 for e in turn_entries if e["speaker"] is not None)
    unassigned = len(turn_entries) - assigned

    return {
        "summary": {
            "total_turns": len(turns),
            "embedded_turns": len(embedded_indices),
            "skipped_short_turns": len(turns) - len(embedded_indices),
            "speaker_count": len(speakers),
            "assigned_turns": assigned,
            "unassigned_turns": unassigned,
        },
        "speakers": speakers,
        "turns": turn_entries,
    }


def main():
    run = start_run("cluster_speakers")
    parser = argparse.ArgumentParser(
        description="Cluster speaker turns by voice embedding."
    )
    parser.add_argument("--words", required=True, help="CTC-aligned words JSON.")
    parser.add_argument("--video", required=True, help="Source video/audio file.")
    parser.add_argument("--output", default="", help="Output speaker map JSON.")
    parser.add_argument(
        "--threshold", type=float, default=0.7,
        help="Cosine distance threshold for anchor clustering (default: 0.7).",
    )
    parser.add_argument(
        "--assign-threshold", type=float, default=0.75,
        help="Max cosine distance for assigning short turns to clusters (default: 0.75).",
    )
    parser.add_argument(
        "--anchor-duration", type=float, default=_ANCHOR_DURATION_S,
        help=f"Minimum turn duration for anchor turns (default: {_ANCHOR_DURATION_S}s).",
    )
    parser.add_argument(
        "--min-turn-duration", type=float, default=_MIN_EMBED_DURATION_S,
        help=f"Skip turns shorter than this (default: {_MIN_EMBED_DURATION_S}s).",
    )
    args = parser.parse_args()

    if not args.output:
        stem = Path(args.words).stem.replace("_ctc_words", "")
        args.output = str(Path(args.words).parent / f"{stem}_speaker_map.json")

    print(f"Loading turns from {args.words}")
    turns = load_turns(args.words)
    print(f"  {len(turns)} turns")

    print(f"Extracting audio...")
    with tempfile.TemporaryDirectory() as work_dir:
        audio_path = os.path.join(work_dir, "audio_16k.wav")
        extract_full_audio(args.video, audio_path)

        print(f"Extracting turn waveforms (min duration: {args.min_turn_duration}s)...")
        waveforms, embedded_indices = extract_turn_waveforms(
            audio_path, turns, min_duration_s=args.min_turn_duration,
        )
        print(f"  {len(waveforms)} turns meet duration threshold "
              f"({len(turns) - len(waveforms)} skipped)")

        if not waveforms:
            print("No turns to embed.")
            return

        print(f"Loading speaker embedding model: {MODEL_SOURCE}")
        model = EncoderClassifier.from_hparams(
            source=MODEL_SOURCE,
            savedir=os.path.join(work_dir, "speechbrain_model"),
            run_opts={"device": DEVICE},
        )

        print(f"Computing embeddings for {len(waveforms)} turns...")
        embeddings = compute_embeddings(model, waveforms)
        print(f"  Embedding shape: {embeddings.shape}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Build duration array for embedded turns
    turn_lookup = {t["turn_index"]: t for t in turns}
    turn_durations = np.array([turn_lookup[idx]["duration"] for idx in embedded_indices])

    print(f"Clustering (anchor threshold={args.threshold}, assign threshold={args.assign_threshold})...")
    labels = cluster_speakers_two_pass(
        embeddings,
        turn_durations,
        anchor_duration_s=args.anchor_duration,
        threshold=args.threshold,
        assign_threshold=args.assign_threshold,
    )
    n_speakers = len(set(labels[labels >= 0]))
    n_unassigned = int(np.sum(labels < 0))
    print(f"  {n_speakers} speaker clusters, {n_unassigned} unassigned turns")

    speaker_map = build_speaker_map(turns, embedded_indices, labels, embeddings)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(speaker_map, f, ensure_ascii=False, indent=2)
    print(f"\nSpeaker map written to {args.output}")

    # Print speaker summary
    for spk_id, stats in speaker_map["speakers"].items():
        print(f"  {spk_id}: {stats['turn_count']} turns, {stats['total_duration_s']}s")

    metadata = finish_run(
        run,
        inputs={"words_json": args.words, "video": args.video},
        outputs={"speaker_map_json": args.output},
        settings={
            "model": MODEL_SOURCE,
            "anchor_threshold": args.threshold,
            "assign_threshold": args.assign_threshold,
            "anchor_duration": args.anchor_duration,
            "min_turn_duration": args.min_turn_duration,
        },
        stats={
            "total_turns": len(turns),
            "embedded_turns": len(embedded_indices),
            "speaker_clusters": n_speakers,
            "unassigned_turns": n_unassigned,
        },
    )
    write_metadata(args.output, metadata)
    print(f"Metadata written: {metadata_path(args.output)}")


if __name__ == "__main__":
    main()
