"""Helpers for writing per-run metadata sidecars."""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

from chigyusubs.paths import find_episode_dir_from_path

RUN_METADATA_SCHEMA_VERSION = 1


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
    payload["metadata_schema_version"] = RUN_METADATA_SCHEMA_VERSION
    payload["run_finished_at"] = now_iso()
    payload["elapsed_seconds"] = round(time.perf_counter() - run["_perf_start"], 3)
    payload.update(extra)
    return payload


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "artifact"


def _run_id(payload: dict) -> str:
    started = str(payload.get("run_started_at", now_iso()))
    started = started.replace(":", "").replace("+", "_plus_")
    started = started.replace(".", "_").replace("-", "").replace("T", "_").replace("Z", "_utc")
    return f"{started}__{_slug(str(payload.get('step', 'run')))}"


def _relative_artifact_slug(output_path: Path, episode_dir: Path | None) -> str:
    if episode_dir is not None:
        try:
            return "__".join(output_path.relative_to(episode_dir).parts)
        except ValueError:
            pass
    return _slug(output_path.name)


def _write_run_ledger(output_path: Path, meta_path: Path, payload: dict) -> None:
    episode_dir = find_episode_dir_from_path(output_path)
    if episode_dir is None:
        return

    run_id = _run_id(payload)
    run_dir = episode_dir / "logs" / "runs" / run_id
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    artifact_slug = _relative_artifact_slug(output_path, episode_dir)
    artifact_meta_path = artifacts_dir / f"{artifact_slug}.meta.json"

    ledger_payload = dict(payload)
    ledger_payload["canonical_output_path"] = str(output_path)
    ledger_payload["canonical_metadata_path"] = str(meta_path)
    ledger_payload["episode_dir"] = str(episode_dir)
    ledger_payload["run_id"] = run_id

    artifact_meta_path.write_text(
        json.dumps(ledger_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_path = run_dir / "run.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "metadata_schema_version": RUN_METADATA_SCHEMA_VERSION,
            "run_id": run_id,
            "step": payload.get("step"),
            "run_started_at": payload.get("run_started_at"),
            "run_finished_at": payload.get("run_finished_at"),
            "elapsed_seconds": payload.get("elapsed_seconds"),
            "episode_dir": str(episode_dir),
            "invocation": payload.get("invocation"),
            "inputs": payload.get("inputs"),
            "outputs": payload.get("outputs"),
            "settings": payload.get("settings"),
            "stats": payload.get("stats"),
            "artifacts": [],
        }

    manifest["metadata_schema_version"] = RUN_METADATA_SCHEMA_VERSION
    manifest["step"] = payload.get("step")
    manifest["run_started_at"] = payload.get("run_started_at")
    manifest["run_finished_at"] = payload.get("run_finished_at")
    manifest["elapsed_seconds"] = payload.get("elapsed_seconds")
    manifest["invocation"] = payload.get("invocation")
    manifest["inputs"] = payload.get("inputs")
    manifest["outputs"] = payload.get("outputs")
    manifest["settings"] = payload.get("settings")
    manifest["stats"] = payload.get("stats")

    artifact_entry = {
        "output_path": str(output_path),
        "metadata_path": str(meta_path),
        "run_artifact_metadata_path": str(artifact_meta_path),
    }
    artifacts = [item for item in manifest.get("artifacts", []) if item.get("output_path") != str(output_path)]
    artifacts.append(artifact_entry)
    manifest["artifacts"] = sorted(artifacts, key=lambda item: item["output_path"])
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_run_readme(run_dir, manifest)


def _yamlish(label: str, value: object) -> list[str]:
    dumped = json.dumps(value, ensure_ascii=False, indent=2)
    lines = dumped.splitlines() or ["null"]
    if len(lines) == 1:
        return [f"{label}: {lines[0]}"]
    out = [f"{label}: |"]
    out.extend(f"  {line}" for line in lines)
    return out


def _write_run_readme(run_dir: Path, manifest: dict) -> None:
    comment_lines = ["<!--"]
    for label in [
        "metadata_schema_version",
        "run_id",
        "step",
        "run_started_at",
        "run_finished_at",
        "elapsed_seconds",
        "episode_dir",
        "invocation",
        "inputs",
        "outputs",
        "settings",
        "stats",
    ]:
        comment_lines.extend(_yamlish(label, manifest.get(label)))
    comment_lines.append("-->")
    artifacts = manifest.get("artifacts", [])
    body = [
        *comment_lines,
        "",
        f"# Run {manifest.get('run_id')}",
        "",
        f"- Step: `{manifest.get('step')}`",
        f"- Started: `{manifest.get('run_started_at')}`",
        f"- Finished: `{manifest.get('run_finished_at')}`",
        f"- Elapsed: `{manifest.get('elapsed_seconds')}` seconds",
        f"- Episode: `{manifest.get('episode_dir')}`",
        "",
        "## Workflow",
        "",
        "Canonical artifacts remain in their episode folders. This run directory is the audit/logging view for the same work.",
        "",
        "## Artifacts",
        "",
    ]
    if artifacts:
        body.extend(f"- `{item['output_path']}`" for item in artifacts)
    else:
        body.append("- None recorded")
    body.append("")
    (run_dir / "README.md").write_text("\n".join(body) + "\n", encoding="utf-8")


def write_metadata(output_path: str | Path, payload: dict):
    output_path = Path(output_path)
    meta_path = metadata_path(output_path)
    meta_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_run_ledger(output_path, meta_path, payload)
