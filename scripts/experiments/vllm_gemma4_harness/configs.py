"""Harness configurations being compared.

Each config is a dict describing a single way to ask the vLLM server to
transcribe a chunk. New configs are added by appending to CONFIGS. Keep
the config name prefix letter-numeric (`A_`, `B_`, ...) so ordering in
result tables stays stable.

Prompt content is intentionally defined here, not inside the runner,
so the git history of this file is the audit trail for "what prompt
produced what numbers".
"""
from __future__ import annotations

from typing import TypedDict


class HarnessConfig(TypedDict, total=False):
    name: str
    frames: bool          # include frames from the spec's frames_dir
    frame_stride: int     # stride across spec frames (1 = all, 2 = every other)
    video: bool           # include video clip via video_url content part
    mst: int | None       # max_soft_tokens to pass via mm_processor_kwargs
    prompt: str           # user-turn instruction text
    system: str           # optional static system-role message (prime goes here)
    system_template: str  # optional per-segment system template — run_bench
                          # fills {oracle_names} etc. before sending
    guided_json: dict     # optional JSON schema — vLLM constrains output to it
    json_field: str       # field name to extract from JSON response as plain text


# --- prompts ------------------------------------------------------------

BASE_PROMPT = (
    "Transcribe the spoken Japanese dialogue in this clip faithfully.\n"
    "Output only plain Japanese text — no translation, no speaker names, "
    "no timestamps."
)

# Priming block assembled from the kind of inputs the pipeline already
# produces per episode: domain kit + glossary + chyron-extracted names.
# This template is episode-specific; the runner can substitute {domain},
# {vocabulary}, {cast} placeholders if a spec file provides them, but a
# default is hard-coded here for the Killah Kuts S01E01 benchmark.
_DEFAULT_DOMAIN = (
    "This is a Japanese variety show called 'Killah Kuts' — a 'Sports Stun "
    "Gun' combat exhibition where comedians fight in a cage wearing stun-gun "
    "gloves. Commentators and coaches describe movement in boxing/MMA "
    "vocabulary."
)

_DEFAULT_VOCAB = [
    "フェイント", "ジャブ", "レフト", "ライト", "フック", "ガード",
    "ノーガード", "サークリング", "ステップ", "フットワーク", "スイッチ",
    "タックル", "キープ", "ケージ", "プレッシャー", "リーチ",
    "レフェリー", "ストップ", "スタンガン", "キャッチャー", "アスリート",
    # NOTE: フェンシング was removed after seg08 regression on 2026-04-11 —
    # when included it out-competed レフェリー in the decoder prior.
    # Only include terms with high confidence they appear verbatim.
]

_DEFAULT_CAST = [
    "みなみかわ", "お見送り芸人しんいち", "大崎", "設楽統",
    "道尾", "道雄", "バナナマン", "アンガールズ",
]

PRIMED_PROMPT = (
    f"{_DEFAULT_DOMAIN}\n"
    f"Expect terms like: {', '.join(_DEFAULT_VOCAB)}.\n"
    f"Fighters and personalities: {', '.join(_DEFAULT_CAST)}.\n\n"
    "Transcribe the spoken Japanese dialogue in this clip faithfully.\n"
    "Output only plain Japanese text — no translation, no speaker names, "
    "no timestamps."
)

# Minimal structured-output schema for guided_json configs. One
# required string field. No additional properties so the decoder can
# never emit a second key (which is how the seg04 cast-list recitation
# leaked out in D — it appended the full list as unquoted continuation).
DIALOGUE_SCHEMA: dict = {
    "type": "object",
    "properties": {"dialogue": {"type": "string"}},
    "required": ["dialogue"],
    "additionalProperties": False,
}


# System-role variant: the domain + vocab + cast block goes in a
# dedicated system message, and the user turn carries only the
# transcription instruction. Hypothesis: this eliminates the seg04
# "recitation" failure where the model parroted the cast list as
# output (the list bleeding into the user-turn instructions behaves
# like a prompt-injection vector). The system message additionally
# marks the vocab/cast as reference-only.
PRIMED_SYSTEM = (
    f"{_DEFAULT_DOMAIN}\n\n"
    f"Reference vocabulary (may appear in dialogue, do NOT output "
    f"unless actually spoken): {', '.join(_DEFAULT_VOCAB)}.\n"
    f"Reference cast names (may appear in dialogue, do NOT output "
    f"unless actually spoken): {', '.join(_DEFAULT_CAST)}."
)

# Oracle-names variant: simulates a perfect OCR pre-pass by telling the
# model exactly which cast names are spoken in THIS specific clip. The
# {oracle_names} placeholder is filled by run_bench from seg["names"]
# (which comes from CTC gold, so this is a cheating upper bound — it
# measures whether E2B can latch onto known-present name strings when
# told they are there). If recall on these clips jumps, the OCR pre-pass
# plan is viable; if not, the audio tower itself has a harder ceiling.
ORACLE_SYSTEM_TEMPLATE = (
    f"{_DEFAULT_DOMAIN}\n\n"
    f"Reference vocabulary (may appear in dialogue, do NOT output "
    f"unless actually spoken): {', '.join(_DEFAULT_VOCAB)}.\n\n"
    "Cast names spoken in THIS SPECIFIC clip (these strings appear in "
    "the audio verbatim — use them exactly as written when "
    "transcribing, do not transliterate to kanji): {oracle_names}."
)


# --- configs ------------------------------------------------------------

CONFIGS: list[HarnessConfig] = [
    {
        "name": "A_audio_base",
        "frames": False,
        "mst": None,
        "prompt": BASE_PROMPT,
    },
    {
        "name": "B_audio_primed",
        "frames": False,
        "mst": None,
        "prompt": PRIMED_PROMPT,
    },
    {
        "name": "C_vision_primed_1fps280",
        "frames": True,
        "frame_stride": 1,
        "mst": 280,
        "prompt": PRIMED_PROMPT,
    },
    {
        "name": "D_audio_sysrole_primed",
        "frames": False,
        "mst": None,
        "system": PRIMED_SYSTEM,
        "prompt": BASE_PROMPT,
    },
    {
        "name": "E_vision_sysrole_primed_1fps280",
        "frames": True,
        "frame_stride": 1,
        "mst": 280,
        "system": PRIMED_SYSTEM,
        "prompt": BASE_PROMPT,
    },
    {
        "name": "F_audio_sysrole_guided_json",
        "frames": False,
        "mst": None,
        "system": PRIMED_SYSTEM,
        "prompt": (
            BASE_PROMPT
            + "\nRespond as JSON: {\"dialogue\": \"<full transcript>\"}."
        ),
        "guided_json": DIALOGUE_SCHEMA,
        "json_field": "dialogue",
    },
    {
        "name": "G_vision_sysrole_guided_json_1fps280",
        "frames": True,
        "frame_stride": 1,
        "mst": 280,
        "system": PRIMED_SYSTEM,
        "prompt": (
            BASE_PROMPT
            + "\nRespond as JSON: {\"dialogue\": \"<full transcript>\"}."
        ),
        "guided_json": DIALOGUE_SCHEMA,
        "json_field": "dialogue",
    },
    {
        "name": "H_audio_sysrole_oracle_names",
        "frames": False,
        "mst": None,
        "system_template": ORACLE_SYSTEM_TEMPLATE,
        "prompt": BASE_PROMPT,
    },
    {
        "name": "I_vision_sysrole_oracle_names_1fps280",
        "frames": True,
        "frame_stride": 1,
        "mst": 280,
        "system_template": ORACLE_SYSTEM_TEMPLATE,
        "prompt": BASE_PROMPT,
    },
    # --- video_url configs (E4B only — E2B hard-codes mst=70) ----------
    {
        "name": "J_video_sysrole_primed",
        "video": True,
        "frames": False,
        "mst": None,
        "system": PRIMED_SYSTEM,
        "prompt": BASE_PROMPT,
    },
    {
        "name": "K_video_sysrole_oracle_names",
        "video": True,
        "frames": False,
        "mst": None,
        "system_template": ORACLE_SYSTEM_TEMPLATE,
        "prompt": BASE_PROMPT,
    },
    # --- high-res vision configs (E4B bf16 only — E2B regresses past mst=280) ----
    {
        "name": "L_vision_sysrole_primed_1fps560",
        "frames": True,
        "frame_stride": 1,
        "mst": 560,
        "system": PRIMED_SYSTEM,
        "prompt": BASE_PROMPT,
    },
    {
        "name": "M_vision_sysrole_primed_1fps1120",
        "frames": True,
        "frame_stride": 1,
        "mst": 1120,
        "system": PRIMED_SYSTEM,
        "prompt": BASE_PROMPT,
    },
    {
        "name": "N_vision_sysrole_oracle_names_1fps560",
        "frames": True,
        "frame_stride": 1,
        "mst": 560,
        "system_template": ORACLE_SYSTEM_TEMPLATE,
        "prompt": BASE_PROMPT,
    },
    {
        "name": "O_vision_sysrole_oracle_names_1fps1120",
        "frames": True,
        "frame_stride": 1,
        "mst": 1120,
        "system_template": ORACLE_SYSTEM_TEMPLATE,
        "prompt": BASE_PROMPT,
    },
]
