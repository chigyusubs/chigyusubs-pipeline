"""Scoring utilities for harness benchmark runs.

Metrics are intentionally simple and explicit — this is for tracking
trends across experiments, not for replacing a real WER/CER eval. If
you need finer measurements, wire up jiwer or the project's existing
`chigyusubs.transcript_comparison` utilities instead.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass
class SegmentScore:
    kata_hits: list[str]
    kata_total: int
    name_hits: list[str]
    name_total: int
    lcs_ratio: float

    @property
    def kata_recall(self) -> float:
        return len(self.kata_hits) / self.kata_total if self.kata_total else 0.0

    @property
    def name_recall(self) -> float:
        return len(self.name_hits) / self.name_total if self.name_total else 0.0

    def as_dict(self) -> dict:
        return {
            "kata_recall": f"{len(self.kata_hits)}/{self.kata_total}",
            "name_recall": f"{len(self.name_hits)}/{self.name_total}",
            "kata_hits": self.kata_hits,
            "name_hits": self.name_hits,
            "lcs_ratio": round(self.lcs_ratio, 3),
        }


def score_segment(
    output: str,
    target: str,
    kata: list[str],
    names: list[str],
) -> SegmentScore:
    """Score one model output against a gold-standard segment.

    kata/name recall is substring presence — no fuzzy matching. If a
    name appears in a transformed form (e.g., しんいち → シイチ) it
    counts as a miss, which is what we want: fuzzy-hits hide real
    degradation in the hardest cases.
    """
    kata_hits = [k for k in kata if k in output]
    name_hits = [n for n in names if n in output]
    lcs = difflib.SequenceMatcher(a=target, b=output).ratio()
    return SegmentScore(
        kata_hits=kata_hits,
        kata_total=len(kata),
        name_hits=name_hits,
        name_total=len(names),
        lcs_ratio=lcs,
    )


def aggregate(rows: list[dict], config_name: str) -> dict:
    """Aggregate a config's per-segment scores into a summary row."""
    kata_num = kata_den = name_num = name_den = 0
    lcs_sum = 0.0
    n = 0
    for row in rows:
        r = row.get(config_name, {})
        if "kata_recall" not in r:
            continue
        k_num, k_den = map(int, r["kata_recall"].split("/"))
        n_num, n_den = map(int, r["name_recall"].split("/"))
        kata_num += k_num
        kata_den += k_den
        name_num += n_num
        name_den += n_den
        lcs_sum += r["lcs_ratio"]
        n += 1
    return {
        "config": config_name,
        "kata_num": kata_num, "kata_den": kata_den,
        "kata_pct": round(100 * kata_num / max(kata_den, 1), 1),
        "name_num": name_num, "name_den": name_den,
        "name_pct": round(100 * name_num / max(name_den, 1), 1),
        "mean_lcs": round(lcs_sum / max(n, 1), 3),
        "n_segments": n,
    }
