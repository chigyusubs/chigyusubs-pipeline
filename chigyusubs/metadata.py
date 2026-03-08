"""Helpers for writing per-run metadata sidecars."""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def metadata_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_name(path.name + ".meta.json")


def start_run(step: str) -> dict:
    return {
        "_perf_start": time.perf_counter(),
        "step": step,
        "run_started_at": now_iso(),
        "invocation": {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
        },
    }


def finish_run(run: dict, **extra) -> dict:
    payload = {k: v for k, v in run.items() if not k.startswith("_")}
    payload["run_finished_at"] = now_iso()
    payload["elapsed_seconds"] = round(time.perf_counter() - run["_perf_start"], 3)
    payload.update(extra)
    return payload


def write_metadata(output_path: str | Path, payload: dict):
    meta_path = metadata_path(output_path)
    meta_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
