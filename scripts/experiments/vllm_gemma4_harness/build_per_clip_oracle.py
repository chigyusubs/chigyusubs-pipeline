#!/usr/bin/env python3
"""Assemble per-clip cast-name oracles from an OCR sweep + glossary.

Pipeline step that turns the existing K_video_sysrole_oracle_names config
from "uses gold names" (cheating upper bound) into "uses OCR-derived
names" (the real production path). For each clip window we ask: which
canonical cast names from the glossary appeared in the on-screen text of
OCR batches whose time range overlaps the clip?

A canonical name "appears" when either the canonical string or any of
its glossary variants is a substring of any line in the batch. Match
on the *clip window* (window_start..window_start+window_dur), not the
narrower seg_start..seg_end — what we want is "which cards/telops were
visible while the model is listening to this clip", and the model sees
the whole window.

Usage (patch a spec in place):
    python3 build_per_clip_oracle.py \\
        --ocr results/killah_kuts_s01e01_ocr_sweep_26bmoe_hip_fps05_20260426.json \\
        --glossary results/killah_kuts_s01e01_glossary_31b_clean_20260415.json \\
        --spec eval_specs/killah_kuts_s01e01_n24.json \\
        --out  eval_specs/killah_kuts_s01e01_n24_oracle.json

Or query a single window:
    python3 build_per_clip_oracle.py \\
        --ocr ... --glossary ... \\
        --t-start 420 --t-end 432
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_glossary_cast(glossary_path: Path) -> dict[str, list[str]]:
    """Return canonical_name -> [surface forms (canonical + variants)].

    Pulls from mc / announcer / referee / creator / fighters. Section
    markers, rule terms, and show_terms are intentionally excluded —
    those aren't names where transliteration help matters.
    """
    g = json.loads(glossary_path.read_text(encoding="utf-8"))
    inner = g.get("glossary", g)

    cast: dict[str, list[str]] = {}

    def push(name: str | None, variants: list[str] | None) -> None:
        if not name:
            return
        forms = {name}
        forms.update([v for v in (variants or []) if v])
        cast[name] = sorted(forms, key=len, reverse=True)

    for role in ("mc", "announcer", "referee"):
        block = inner.get(role) or {}
        push(block.get("name"), block.get("variants"))

    creator = inner.get("creator")
    if creator:
        cast[creator] = [creator]

    for f in inner.get("fighters", []) or []:
        push(f.get("name"), f.get("variants"))

    return cast


def index_ocr_batches(
    ocr_path: Path, cast: dict[str, list[str]],
) -> list[dict]:
    """For each OCR batch, record which canonicals were seen.

    Returns a list of {t_start, t_end, canonicals: set[str]} dicts in
    batch order. canonicals is a frozenset of canonical names whose
    canonical OR any variant string was a substring of the batch's
    line blob.
    """
    sweep = json.loads(ocr_path.read_text(encoding="utf-8"))
    indexed: list[dict] = []
    for b in sweep.get("batches", []):
        if "error" in b:
            continue
        blob = "\n".join(b.get("lines") or [])
        seen: set[str] = set()
        for canonical, surfaces in cast.items():
            for s in surfaces:
                if s and s in blob:
                    seen.add(canonical)
                    break
        indexed.append({
            "batch_idx": b.get("batch_idx"),
            "t_start": float(b.get("t_start", 0.0)),
            "t_end": float(b.get("t_end", 0.0)),
            "canonicals": seen,
        })
    return indexed


def oracle_for_window(
    indexed: list[dict], t_start: float, t_end: float,
) -> list[str]:
    """Union canonicals across all OCR batches overlapping [t_start, t_end].

    Order: by first batch (earliest t_start) where each canonical
    appeared. This way the oracle list reads as "cards roughly in the
    order they came up", which is harmless to the model but pleasant to
    eyeball when debugging.
    """
    first_seen: dict[str, float] = {}
    for b in indexed:
        if b["t_end"] < t_start or b["t_start"] > t_end:
            continue
        for c in b["canonicals"]:
            if c not in first_seen:
                first_seen[c] = b["t_start"]
    return [c for c, _ in sorted(first_seen.items(), key=lambda kv: kv[1])]


def patch_spec(
    spec_path: Path, indexed: list[dict], out_path: Path,
) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    counts: list[int] = []
    for seg in spec["segments"]:
        w_start = float(seg.get("window_start", seg["seg_start"]))
        w_dur = float(seg.get("window_dur", 0.0))
        w_end = w_start + w_dur if w_dur else float(seg["seg_end"])
        names = oracle_for_window(indexed, w_start, w_end)
        seg["oracle_names"] = names
        counts.append(len(names))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "n_segments": len(spec["segments"]),
        "mean_oracle_size": round(sum(counts) / max(len(counts), 1), 2),
        "max_oracle_size": max(counts) if counts else 0,
        "empty_oracle_segments": sum(1 for c in counts if c == 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", required=True,
                    help="OCR sweep JSON (output of ocr_sweep.py)")
    ap.add_argument("--glossary", required=True,
                    help="Curated glossary JSON (output of build_glossary.py)")
    ap.add_argument("--spec", default="",
                    help="Eval spec to patch with per-segment oracle_names. "
                         "If omitted, --t-start/--t-end must be given.")
    ap.add_argument("--out", default="",
                    help="Output spec path. Required with --spec.")
    ap.add_argument("--t-start", type=float, default=None)
    ap.add_argument("--t-end", type=float, default=None)
    args = ap.parse_args()

    cast = load_glossary_cast(Path(args.glossary))
    if not cast:
        raise SystemExit("glossary has no cast names — nothing to match against")
    indexed = index_ocr_batches(Path(args.ocr), cast)

    print(f"Cast canonicals:    {len(cast)}  ({list(cast.keys())})")
    print(f"OCR batches indexed: {len(indexed)}")
    nonempty = sum(1 for b in indexed if b["canonicals"])
    print(f"  with cast hits:    {nonempty}")

    if args.spec:
        if not args.out:
            raise SystemExit("--out is required with --spec")
        stats = patch_spec(Path(args.spec), indexed, Path(args.out))
        print(f"\nWrote {args.out}")
        print(f"  n_segments:             {stats['n_segments']}")
        print(f"  mean oracle size:       {stats['mean_oracle_size']}")
        print(f"  max oracle size:        {stats['max_oracle_size']}")
        print(f"  empty oracle segments:  {stats['empty_oracle_segments']}")
    elif args.t_start is not None and args.t_end is not None:
        names = oracle_for_window(indexed, args.t_start, args.t_end)
        print(f"\nOracle for [{args.t_start}, {args.t_end}]:")
        if names:
            for n in names:
                print(f"  - {n}")
        else:
            print("  (none)")
    else:
        raise SystemExit("pass either --spec/--out or --t-start/--t-end")


if __name__ == "__main__":
    main()
