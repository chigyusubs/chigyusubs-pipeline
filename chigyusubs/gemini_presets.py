"""Named Gemini settings presets for maintained workflows."""

from __future__ import annotations

from typing import Any


SCRIPT_DEFAULTS: dict[str, dict[str, Any]] = {
    "transcribe_gemini_video": {
        "model": "gemini-2.5-pro",
        "temperature": 0.1,
        "retry_temperature": 0.3,
        "spoken_only": False,
        "media_resolution": "unspecified",
        "thinking_level": "unspecified",
        "thinking_budget": None,
        "rolling_context_chunks": 1,
        "max_request_retries": 8,
        "max_timeout_errors": 3,
        "max_rate_limit_errors": 4,
    },
    "extract_gemini_chunk_ocr": {
        "model": "gemini-3.1-flash-lite-preview",
        "temperature": 0.0,
        "media_resolution": "high",
        "thinking_level": "high",
        "thinking_budget": None,
    },
}


PRESETS: dict[str, dict[str, Any]] = {
    "flash25_free_default": {
        "script": "transcribe_gemini_video",
        "description": "Maintained free-tier transcript default for 2.5-Flash: video spoken-only high-res with no thinking override.",
        "settings": {
            "model": "gemini-2.5-flash",
            "temperature": 0.0,
            "retry_temperature": 0.3,
            "spoken_only": True,
            "media_resolution": "high",
            "thinking_level": "unspecified",
            "thinking_budget": None,
        },
    },
    "flash_free_default": {
        "script": "transcribe_gemini_video",
        "description": "Maintained free-tier transcript default: 3-Flash video spoken-only low-thinking high-res.",
        "settings": {
            "model": "gemini-3-flash-preview",
            "temperature": 0.0,
            "retry_temperature": 0.3,
            "spoken_only": True,
            "media_resolution": "high",
            "thinking_level": "low",
            "thinking_budget": None,
        },
    },
    "flash_visual_artifact": {
        "script": "transcribe_gemini_video",
        "description": "3-Flash video transcript with selective [画面: ...] visual cues retained.",
        "settings": {
            "model": "gemini-3-flash-preview",
            "temperature": 0.0,
            "retry_temperature": 0.3,
            "spoken_only": False,
            "media_resolution": "high",
            "thinking_level": "low",
            "thinking_budget": None,
        },
    },
    "flashlite_debug_transcript": {
        "script": "transcribe_gemini_video",
        "description": "Maintained Flash-Lite debug transcript preset: spoken-only high-res with no rolling context and bounded retries.",
        "settings": {
            "model": "gemini-3.1-flash-lite-preview",
            "temperature": 0.0,
            "retry_temperature": 0.3,
            "spoken_only": True,
            "media_resolution": "high",
            "thinking_level": "unspecified",
            "thinking_budget": None,
            "rolling_context_chunks": 0,
            "max_request_retries": 4,
            "max_timeout_errors": 2,
            "max_rate_limit_errors": 2,
        },
    },
    "pro_quality_video": {
        "script": "transcribe_gemini_video",
        "description": "Higher-quality Pro video transcript baseline.",
        "settings": {
            "model": "gemini-3.1-pro-preview",
            "temperature": 0.0,
            "retry_temperature": 0.2,
            "spoken_only": True,
            "media_resolution": "high",
            "thinking_level": "low",
            "thinking_budget": None,
        },
    },
    "flashlite_ocr_sidecar": {
        "script": "extract_gemini_chunk_ocr",
        "description": "Maintained Flash-Lite OCR sidecar preset for chunkwise visible text extraction.",
        "settings": {
            "model": "gemini-3.1-flash-lite-preview",
            "temperature": 0.0,
            "media_resolution": "high",
            "thinking_level": "high",
            "thinking_budget": None,
        },
    },
}


def preset_names(script_name: str) -> list[str]:
    return sorted(name for name, spec in PRESETS.items() if spec["script"] == script_name)


def resolve_settings(
    script_name: str,
    preset_name: str | None,
    overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str | None]:
    if script_name not in SCRIPT_DEFAULTS:
        raise KeyError(f"Unknown script preset namespace: {script_name}")
    resolved = dict(SCRIPT_DEFAULTS[script_name])
    chosen = None
    if preset_name:
        spec = PRESETS.get(preset_name)
        if spec is None:
            raise KeyError(f"Unknown Gemini preset: {preset_name}")
        if spec["script"] != script_name:
            raise ValueError(f"Preset {preset_name} is not valid for {script_name}")
        resolved.update(spec["settings"])
        chosen = preset_name
    for key, value in (overrides or {}).items():
        if value is not None:
            resolved[key] = value
    return resolved, chosen
