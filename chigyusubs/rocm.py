"""ROCm environment helpers for CTranslate2/faster-whisper paths."""

from __future__ import annotations

import os
from pathlib import Path


def ensure_rocm_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return an env mapping with the repo's required ROCm defaults."""
    result = dict(env or os.environ)
    result.setdefault("CT2_CUDA_ALLOCATOR", "cub_caching")
    rocm_lib = Path("/opt/rocm-7.2.0/lib")
    if rocm_lib.exists():
        current = result.get("LD_LIBRARY_PATH", "")
        entries = [item for item in current.split(":") if item]
        if str(rocm_lib) not in entries:
            result["LD_LIBRARY_PATH"] = ":".join([str(rocm_lib)] + entries) if entries else str(rocm_lib)
    return result


def apply_rocm_env() -> None:
    """Mutate os.environ in-process so imported ROCm libs see the expected env."""
    # Codex note: this helper normalizes env vars, but sandboxed runs still need
    # an escalated command to see the ROCm device, and LD_LIBRARY_PATH must be
    # present when Python starts for ctranslate2/faster-whisper to load cleanly.
    os.environ.update(ensure_rocm_env())
