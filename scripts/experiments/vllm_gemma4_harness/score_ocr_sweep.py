#!/usr/bin/env python3
"""Score one or more OCR sweep result files against a gold list.

The gold list is a flat list of expected on-screen strings, grouped by
category (cast names, section markers, rule terms, show title). For
each sweep result we compute substring recall per category.

Strict substring match by design — same convention as scoring.py.

CAVEAT: when the gold is derived from a glossary that was itself built
from one of the sweeps under comparison (via --gold-from-glossary),
the source-of-gold sweep will trivially score ~100%. The useful signal
in that mode is what each *other* sweep missed relative to the chosen
oracle — not the absolute pct. For a fair head-to-head, supply a gold
spec curated independently (or built from a different model) via
--gold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_gold(path: Path) -> dict[str, list[str]]:
    """Load a gold spec.

    The spec is a JSON file with category -> list[str], plus an optional
    `aliases` map from canonical -> [acceptable surface forms].
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw


def recall(text_blob: str, terms: list[str]) -> tuple[list[str], list[str]]:
    """Return (hits, misses) for substring presence in `text_blob`."""
    hits = [t for t in terms if t and t in text_blob]
    misses = [t for t in terms if t and t not in text_blob]
    return hits, misses


def score_sweep(sweep_path: Path, gold: dict) -> dict:
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    lines: list[str] = sweep.get("unique_lines", [])
    blob = "\n".join(lines)

    categories = {
        k: v for k, v in gold.items()
        if isinstance(v, list) and k != "aliases"
    }
    aliases: dict[str, list[str]] = gold.get("aliases", {}) or {}

    per_cat: dict[str, dict] = {}
    overall_num = overall_den = 0
    for cat, terms in categories.items():
        hits, misses = recall(blob, terms)
        # An aliased miss still counts as a hit if any alias surface form
        # appears. Keeps us honest about kana-vs-kanji variation while
        # making it visible in the report.
        rescued: list[tuple[str, str]] = []
        true_misses: list[str] = []
        for m in misses:
            alts = aliases.get(m, [])
            seen_alt = next((a for a in alts if a and a in blob), None)
            if seen_alt:
                rescued.append((m, seen_alt))
            else:
                true_misses.append(m)
        num = len(hits) + len(rescued)
        den = len(terms)
        per_cat[cat] = {
            "recall": f"{num}/{den}",
            "pct": round(100 * num / max(den, 1), 1),
            "hits": hits,
            "rescued_via_alias": [
                {"canonical": c, "via": a} for c, a in rescued
            ],
            "misses": true_misses,
        }
        overall_num += num
        overall_den += den

    return {
        "sweep_file": str(sweep_path),
        "model": sweep.get("model"),
        "fps": sweep.get("fps"),
        "mst": sweep.get("mst"),
        "backend": sweep.get("backend"),
        "n_unique_lines": len(lines),
        "total_wall_s": sweep.get("total_wall"),
        "overall": {
            "recall": f"{overall_num}/{overall_den}",
            "pct": round(100 * overall_num / max(overall_den, 1), 1),
        },
        "categories": per_cat,
    }


def gold_from_glossary(glossary_path: Path) -> dict[str, list[str]]:
    """Derive a flat gold spec from a curated glossary JSON.

    The glossary is the output of build_glossary.py finalize. We treat
    its canonical strings as the gold and its `variants` lists as
    acceptable aliases, so a sweep that produced a misread (`スポーツ
    タンガン` for `スポーツスタンガン`) still records the canonical
    miss but is annotated as rescued via alias.
    """
    g = json.loads(glossary_path.read_text(encoding="utf-8"))
    inner = g.get("glossary", g)

    cast: list[str] = []
    aliases: dict[str, list[str]] = {}

    def push(name: str | None, variants: list[str] | None) -> None:
        if not name:
            return
        cast.append(name)
        if variants:
            aliases[name] = list(variants)

    show_title = inner.get("show_title") or {}
    push(show_title.get("canonical"), show_title.get("variants"))

    for role in ("mc", "announcer", "referee"):
        block = inner.get(role) or {}
        push(block.get("name"), block.get("variants"))

    for f in inner.get("fighters", []) or []:
        push(f.get("name"), f.get("variants"))

    creator = inner.get("creator")
    if creator:
        cast.append(creator)
    em = inner.get("episode_marker")
    section_markers = list(inner.get("section_markers") or [])
    rule_terms = list(inner.get("rule_terms") or [])
    show_terms = list(inner.get("show_terms") or [])

    spec: dict[str, list[str]] = {
        "cast": cast,
        "section_markers": section_markers,
        "rule_terms": rule_terms,
        "show_terms": show_terms,
    }
    if em:
        spec["episode_marker"] = [em]
    return {"aliases": aliases, **spec}


def render_report(reports: list[dict]) -> str:
    cats = ["cast", "section_markers", "rule_terms", "show_terms",
            "episode_marker"]
    lines: list[str] = []
    lines.append(
        f"{'model':<60} {'wall':>7} {'lines':>6}  "
        + "  ".join(f"{c:>16}" for c in cats)
        + "    overall"
    )
    for r in reports:
        model = (r.get("model") or "?")[-58:]
        wall = r.get("total_wall_s")
        wall_s = f"{wall:.0f}s" if wall is not None else "  ?"
        nl = r.get("n_unique_lines", 0)
        cells = []
        for c in cats:
            block = r["categories"].get(c)
            cells.append(block["recall"] if block else "      -")
        lines.append(
            f"{model:<60} {wall_s:>7} {nl:>6}  "
            + "  ".join(f"{cell:>16}" for cell in cells)
            + f"    {r['overall']['recall']} ({r['overall']['pct']}%)"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="",
                    help="JSON file with category -> list[str] gold spec. "
                         "Mutually exclusive with --gold-from-glossary.")
    ap.add_argument("--gold-from-glossary", default="",
                    help="Path to a curated glossary JSON; the canonical "
                         "names + variants will be used as gold.")
    ap.add_argument("--out", default="",
                    help="Optional JSON output path for the structured "
                         "report (per-sweep cat breakdowns).")
    ap.add_argument("sweeps", nargs="+",
                    help="One or more OCR sweep result JSON files.")
    args = ap.parse_args()

    if bool(args.gold) == bool(args.gold_from_glossary):
        raise SystemExit(
            "Pass exactly one of --gold or --gold-from-glossary."
        )
    if args.gold:
        gold = load_gold(Path(args.gold))
    else:
        gold = gold_from_glossary(Path(args.gold_from_glossary))

    reports = [score_sweep(Path(p), gold) for p in args.sweeps]
    print(render_report(reports))

    print()
    print("--- per-category misses ---")
    for r in reports:
        print(f"\n{r['model']} ({r['n_unique_lines']} lines, "
              f"{r['total_wall_s']}s wall)")
        for cat, block in r["categories"].items():
            if block["misses"] or block["rescued_via_alias"]:
                print(f"  [{cat}]")
                for m in block["misses"]:
                    print(f"    miss:    {m!r}")
                for ra in block["rescued_via_alias"]:
                    print(f"    rescued: {ra['canonical']!r} via {ra['via']!r}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {"gold": gold, "reports": reports},
                ensure_ascii=False, indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
