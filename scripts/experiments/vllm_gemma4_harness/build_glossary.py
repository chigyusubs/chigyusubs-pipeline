#!/usr/bin/env python3
"""OCR → glossary pipeline, three interactive stages.

Stages:
  1. discover  — LLM reads OCR telops, emits a skeleton glossary and a
                 list of clarifying questions (with OCR-grounded options)
                 to resolve ambiguities.
  2. clarify   — pure TTY step: present each question, user picks a
                 numbered option or types free text; answers written
                 back into the discovery file.
  3. finalize  — LLM re-reads OCR + discovery + answers, emits the final
                 structured glossary. Substring-validated against OCR to
                 catch rule_term / section_marker hallucinations.

The `all` stage runs discover → clarify → finalize in one go. Each
stage checkpoints to disk so you can resume or re-run any stage in
isolation (e.g. tweak answers, re-finalize).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "google/gemma-4-E4B-it"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1/chat/completions"


# --- Stage 1: DISCOVER --------------------------------------------------

DISCOVER_SYSTEM = (
    "You prepare reference glossaries for Japanese variety shows from "
    "raw OCR telops pulled from a whole episode. This is the DISCOVERY "
    "pass — your job is to propose an initial entity skeleton AND to "
    "surface any ambiguities to a human reviewer as clarifying "
    "questions.\n"
    "\n"
    "Entity roles to identify:\n"
    "- show_title: the brand / title. This is typically an English or "
    "romanized logo (e.g. a phrase in Latin letters on the opening "
    "card). A Japanese subtitle next to it is usually the genre or "
    "tagline, not the title.\n"
    "- episode_marker (e.g. 'EPISODE 1')\n"
    "- mc, announcer, referee, creator, narrator\n"
    "- fighters (plural) — each with the team/affiliation shown in "
    "parentheses next to the name, if any\n"
    "- section_markers — tournament-bracket labels like '1回戦', "
    "'準決勝', '決勝戦'\n"
    "- rule_terms — the show's rulebook lines (actual chyrons that "
    "explain how the game works). NOT your prior knowledge about the "
    "sport; only phrases physically printed on screen.\n"
    "\n"
    "For each entity you list, include `supporting_lines`: 1-3 "
    "VERBATIM quotes from the input that support this entity. Do not "
    "paraphrase.\n"
    "\n"
    "For each clarifying question, include `options` sourced ONLY from "
    "OCR evidence — never invent options. Raise a question when:\n"
    "- The same name appears with 2+ plausibly-different kanji readings "
    "(e.g. 梅木/楠木/極木).\n"
    "- You cannot tell whether a string is the brand title vs a genre "
    "subtitle vs a generic show term.\n"
    "- A fighter name could plausibly be either of two forms "
    "(e.g. 谷拓哉 vs 谷口拓哉) and the input isn't decisive.\n"
    "- A string looks like it could be a section marker / rule term "
    "but is only observed once and might be decorative.\n"
    "\n"
    "Do NOT ask cosmetic questions. Aim for 3-8 focused questions that "
    "actually change the final glossary."
)

DISCOVER_USER_TEMPLATE = (
    "Here are the {n_lines} unique on-screen text blocks from "
    "episode `{episode}`:\n\n"
    "<telops>\n{telops}\n</telops>\n\n"
    "Produce the discovery JSON."
)

DISCOVER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "show_title_candidates": {
            "type": "array", "items": {"type": "string"},
            "description": "Strings that could be the canonical title.",
        },
        "episode_marker_candidates": {
            "type": "array", "items": {"type": "string"},
        },
        "mc_candidates": {"type": "array", "items": {"type": "string"}},
        "announcer_candidates": {"type": "array", "items": {"type": "string"}},
        "referee_candidates": {"type": "array", "items": {"type": "string"}},
        "creator_candidates": {"type": "array", "items": {"type": "string"}},
        "narrator_candidates": {"type": "array", "items": {"type": "string"}},
        "fighter_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "best_guess_name": {"type": "string"},
                    "team": {"type": ["string", "null"]},
                    "variants": {"type": "array", "items": {"type": "string"}},
                    "supporting_lines": {
                        "type": "array", "items": {"type": "string"},
                    },
                },
                "required": ["best_guess_name", "variants",
                             "supporting_lines"],
                "additionalProperties": False,
            },
        },
        "section_marker_candidates": {
            "type": "array", "items": {"type": "string"},
        },
        "rule_term_candidates": {
            "type": "array", "items": {"type": "string"},
        },
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "topic": {
                        "type": "string",
                        "description": (
                            "Short tag, e.g. 'show_title', 'referee', "
                            "'fighter:谷口拓哉'."
                        ),
                    },
                    "question": {"type": "string"},
                    "options": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "OCR-grounded answer choices. Each must "
                            "appear as a substring in the telops."
                        ),
                    },
                    "suggestion": {"type": ["string", "null"]},
                    "supporting_lines": {
                        "type": "array", "items": {"type": "string"},
                    },
                },
                "required": ["id", "topic", "question", "options",
                             "supporting_lines"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "show_title_candidates", "episode_marker_candidates",
        "mc_candidates", "announcer_candidates", "referee_candidates",
        "creator_candidates", "narrator_candidates", "fighter_candidates",
        "section_marker_candidates", "rule_term_candidates", "questions",
    ],
    "additionalProperties": False,
}


# --- Stage 3: FINALIZE --------------------------------------------------

FINALIZE_SYSTEM = (
    "You are completing a reference glossary for a Japanese variety "
    "show. A prior discovery pass extracted candidate entities from "
    "OCR telops and flagged ambiguities. A human reviewer has answered "
    "those questions. Your job now is to produce the FINAL structured "
    "glossary, honoring the reviewer's answers and extracting fighter "
    "metadata (age, height, weight, birthdate) from telop lines that "
    "contain them.\n"
    "\n"
    "Hard rules:\n"
    "- Every string in `rule_terms`, `section_markers`, `show_terms` "
    "must appear as a SUBSTRING of at least one input OCR line. Do "
    "not invent, translate, or paraphrase.\n"
    "- When the reviewer's answer conflicts with your discovery guess, "
    "the reviewer wins.\n"
    "- Fighter-card OCR lines follow the pattern "
    "`<team><name>(<age>) <cm>cm <kg>kg` — when such a line exists "
    "for a fighter, extract age / height_cm / weight_kg as integers.\n"
    "- `birthdate` comes from lines like '1976年5月27日生' — copy them "
    "verbatim.\n"
    "- Variants: keep all the OCR misreadings you gathered so the ASR "
    "downstream can learn from them."
)

FINALIZE_USER_TEMPLATE = (
    "Episode: {episode}\n"
    "\n"
    "OCR telops ({n_lines} lines):\n"
    "<telops>\n{telops}\n</telops>\n"
    "\n"
    "Discovery skeleton:\n"
    "<discovery>\n{discovery}\n</discovery>\n"
    "\n"
    "Reviewer answers:\n"
    "<answers>\n{answers}\n</answers>\n"
    "\n"
    "Produce the final glossary JSON now."
)

GLOSSARY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "show_title": {
            "type": "object",
            "properties": {
                "canonical": {"type": "string"},
                "variants": {"type": "array", "items": {"type": "string"}},
                "romaji": {"type": ["string", "null"]},
            },
            "required": ["canonical", "variants"],
            "additionalProperties": False,
        },
        "episode_marker": {"type": ["string", "null"]},
        "creator": {"type": ["string", "null"]},
        "narrator": {"type": ["string", "null"]},
        "mc": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "team": {"type": ["string", "null"]},
                "variants": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "variants"],
            "additionalProperties": False,
        },
        "announcer": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "variants": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "variants"],
            "additionalProperties": False,
        },
        "referee": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "variants": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "variants"],
            "additionalProperties": False,
        },
        "fighters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "team": {"type": ["string", "null"]},
                    "age": {"type": ["integer", "null"]},
                    "height_cm": {"type": ["integer", "null"]},
                    "weight_kg": {"type": ["integer", "null"]},
                    "birthdate": {"type": ["string", "null"]},
                    "variants": {
                        "type": "array", "items": {"type": "string"},
                    },
                },
                "required": ["name", "variants"],
                "additionalProperties": False,
            },
        },
        "section_markers": {"type": "array", "items": {"type": "string"}},
        "rule_terms": {"type": "array", "items": {"type": "string"}},
        "show_terms": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "show_title", "episode_marker", "creator", "narrator", "mc",
        "announcer", "referee", "fighters", "section_markers",
        "rule_terms", "show_terms",
    ],
    "additionalProperties": False,
}


# --- helpers ------------------------------------------------------------

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE,
)


def extract_json(text: str) -> dict:
    s = text.strip()
    m = _JSON_FENCE_RE.search(s)
    if m:
        s = m.group(1)
    return json.loads(s)


def post(base_url: str, payload: dict,
         timeout: int = 600) -> tuple[dict, float]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url, data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        return body, time.monotonic() - t0
    except urllib.error.HTTPError as e:
        return ({"error": f"HTTP {e.code}",
                 "body": e.read().decode("utf-8", "replace")[:1000]},
                time.monotonic() - t0)


def apply_schema(payload: dict, schema: dict, backend: str,
                 schema_name: str) -> None:
    if backend == "vllm":
        payload["guided_json"] = schema
    else:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name, "schema": schema, "strict": True,
            },
        }
        # Gemma 4 ships with a thinking channel in its chat template;
        # disable it or the model burns the whole budget on reasoning.
        payload["chat_template_kwargs"] = {"enable_thinking": False}


def call_llm(*, base_url: str, model: str, backend: str,
             system: str, user: str, schema: dict, schema_name: str,
             max_tokens: int) -> tuple[dict, float, dict]:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    apply_schema(payload, schema, backend, schema_name)
    body, wall = post(base_url, payload)
    if "error" in body:
        raise SystemExit(
            f"server error: {body['error']}\n{body.get('body', '')}"
        )
    raw = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    try:
        obj = extract_json(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"failed to parse JSON ({e}); raw: {raw[:500]!r}")
    return obj, wall, usage


def load_ocr(path: Path) -> tuple[str, list[str]]:
    ocr = json.loads(path.read_text(encoding="utf-8"))
    return ocr.get("episode", "unknown"), ocr["unique_lines"]


def derive_discovery_path(out: Path) -> Path:
    return out.with_suffix(".discovery.json")


# --- stage implementations ----------------------------------------------

def stage_discover(*, ocr_path: Path, discovery_path: Path, args) -> dict:
    episode, lines = load_ocr(ocr_path)
    telops = "\n".join(lines)
    user = DISCOVER_USER_TEMPLATE.format(
        n_lines=len(lines), episode=episode, telops=telops,
    )
    print(f"[discover] episode={episode}  lines={len(lines)}"
          f"  model={args.model}")
    obj, wall, usage = call_llm(
        base_url=args.base_url, model=args.model, backend=args.backend,
        system=DISCOVER_SYSTEM, user=user, schema=DISCOVER_SCHEMA,
        schema_name="discovery", max_tokens=args.max_tokens,
    )
    print(f"  wall={wall:.1f}s  prompt_tok={usage.get('prompt_tokens')}"
          f"  completion_tok={usage.get('completion_tokens')}"
          f"  questions={len(obj.get('questions', []))}")
    out = {
        "episode": episode,
        "source_ocr": str(ocr_path.resolve()),
        "model": args.model,
        "wall": round(wall, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "discovery": obj,
        "answers": {},
    }
    discovery_path.parent.mkdir(parents=True, exist_ok=True)
    discovery_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"  wrote {discovery_path}")
    return out


def _prompt(msg: str) -> str:
    sys.stdout.write(msg)
    sys.stdout.flush()
    return sys.stdin.readline().rstrip("\n")


def stage_clarify(*, discovery_path: Path) -> dict:
    doc = json.loads(discovery_path.read_text(encoding="utf-8"))
    questions = doc["discovery"].get("questions", [])
    answers: dict[str, str] = dict(doc.get("answers") or {})
    if not questions:
        print("[clarify] no questions raised; nothing to do.")
        return doc

    print(f"\n[clarify] {len(questions)} question(s). "
          "For each: type the number of your choice, or free-text to "
          "override, or blank to accept the suggestion, or 's' to skip.\n")
    for i, q in enumerate(questions, 1):
        qid = q.get("id") or f"q{i}"
        if qid in answers and answers[qid]:
            print(f"--- Q{i}/{len(questions)}  [{qid}]  "
                  f"(already answered: {answers[qid]!r}, keeping) ---")
            continue
        print(f"--- Q{i}/{len(questions)}  [{q.get('topic', qid)}] ---")
        print(f"  {q['question']}")
        for j, sup in enumerate(q.get("supporting_lines") or [], 1):
            print(f"    ctx{j}: {sup}")
        options = q.get("options") or []
        for j, opt in enumerate(options, 1):
            tag = " (suggested)" if opt == q.get("suggestion") else ""
            print(f"    [{j}] {opt}{tag}")
        suggestion = q.get("suggestion")
        default_hint = f" [blank={suggestion!r}]" if suggestion else ""
        raw = _prompt(f"  > your answer{default_hint}: ").strip()
        if raw.lower() == "s":
            print("    (skipped)\n")
            answers[qid] = ""
            continue
        if not raw and suggestion:
            answers[qid] = suggestion
            print(f"    -> {suggestion}\n")
            continue
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                answers[qid] = options[idx]
                print(f"    -> {options[idx]}\n")
                continue
        answers[qid] = raw
        print(f"    -> {raw}\n")

    doc["answers"] = answers
    discovery_path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[clarify] wrote answers to {discovery_path}")
    return doc


def _substring_check(glossary: dict, lines: list[str]) -> list[str]:
    """Return list of strings in rule_terms / section_markers / "
    show_terms that do not appear as a substring of any OCR line.
    """
    text = "\n".join(lines)
    bad: list[str] = []
    for key in ("rule_terms", "section_markers", "show_terms"):
        for s in glossary.get(key, []):
            if s not in text:
                bad.append(f"{key}: {s!r}")
    return bad


def stage_finalize(*, ocr_path: Path, discovery_path: Path,
                   out_path: Path, args) -> dict:
    episode, lines = load_ocr(ocr_path)
    telops = "\n".join(lines)
    doc = json.loads(discovery_path.read_text(encoding="utf-8"))
    discovery = doc["discovery"]
    answers = doc.get("answers") or {}
    answers_rendered = "\n".join(
        f"- [{qid}] {v}" for qid, v in answers.items() if v
    ) or "(no answers provided)"

    user = FINALIZE_USER_TEMPLATE.format(
        episode=episode, n_lines=len(lines), telops=telops,
        discovery=json.dumps(discovery, ensure_ascii=False, indent=2),
        answers=answers_rendered,
    )
    print(f"[finalize] episode={episode}  answers={len(answers)}"
          f"  model={args.model}")
    obj, wall, usage = call_llm(
        base_url=args.base_url, model=args.model, backend=args.backend,
        system=FINALIZE_SYSTEM, user=user, schema=GLOSSARY_SCHEMA,
        schema_name="glossary", max_tokens=args.max_tokens,
    )
    print(f"  wall={wall:.1f}s  prompt_tok={usage.get('prompt_tokens')}"
          f"  completion_tok={usage.get('completion_tokens')}")

    bad = _substring_check(obj, lines)
    if bad:
        print(f"[finalize] WARNING {len(bad)} ungrounded term(s):")
        for b in bad:
            print(f"    {b}")

    out = {
        "episode": episode,
        "source_ocr": str(ocr_path.resolve()),
        "source_discovery": str(discovery_path.resolve()),
        "model": args.model,
        "wall": round(wall, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "ungrounded_terms": bad,
        "glossary": obj,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"  wrote {out_path}")

    g = obj
    empty: dict = {}
    print("\n--- final glossary ---")
    print(f"  show_title:   {g.get('show_title', empty).get('canonical')}")
    print(f"  episode:      {g.get('episode_marker')}")
    print(f"  creator:      {g.get('creator')}")
    print(f"  narrator:     {g.get('narrator')}")
    print(f"  mc:           {g.get('mc', empty).get('name')}  "
          f"({g.get('mc', empty).get('team')})")
    print(f"  announcer:    {g.get('announcer', empty).get('name')}")
    print(f"  referee:      {g.get('referee', empty).get('name')}")
    print(f"  fighters:     {len(g.get('fighters', []))}")
    for f in g.get("fighters", []):
        print(f"    - {f['name']}  team={f.get('team')}  "
              f"age={f.get('age')}  {f.get('height_cm')}cm "
              f"{f.get('weight_kg')}kg  "
              f"birth={f.get('birthdate')}")
    print(f"  sections:     {g.get('section_markers')}")
    print(f"  rule_terms:   {len(g.get('rule_terms', []))}")
    print(f"  show_terms:   {len(g.get('show_terms', []))}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="OCR sweep JSON (output of ocr_sweep.py)")
    ap.add_argument("--out", required=True, help="final glossary JSON path")
    ap.add_argument("--discovery", default="",
                    help="discovery checkpoint path "
                         "(default: <out>.discovery.json)")
    ap.add_argument("--stage",
                    choices=["discover", "clarify", "finalize", "all"],
                    default="all")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--backend",
                    choices=["vllm", "llama-cpp"], default="vllm")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    ocr_path = Path(args.input).resolve()
    out_path = Path(args.out).resolve()
    discovery_path = (Path(args.discovery).resolve() if args.discovery
                      else derive_discovery_path(out_path))

    if args.stage in ("discover", "all"):
        stage_discover(
            ocr_path=ocr_path, discovery_path=discovery_path, args=args,
        )
    if args.stage in ("clarify", "all"):
        stage_clarify(discovery_path=discovery_path)
    if args.stage in ("finalize", "all"):
        stage_finalize(
            ocr_path=ocr_path, discovery_path=discovery_path,
            out_path=out_path, args=args,
        )


if __name__ == "__main__":
    main()
