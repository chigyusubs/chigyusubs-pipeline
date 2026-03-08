# Codex Skills

This directory is the canonical repo-tracked source for project-specific Codex
skills.

Current tracked skills:

- `subtitle-reflow`
- `subtitle-translation`

These are copied into the live Codex home when needed. The live install target
is usually:

```text
~/.codex/skills/
```

Use the repo helper to install or refresh them:

```bash
python3 scripts/install_codex_skills.py
python3 scripts/install_codex_skills.py --skill subtitle-reflow
```

Rules:

- edit skills here, not in `~/.codex/skills/`
- do not copy `.system` skills into the repo
- keep nested assets (`references/`, `agents/`) tracked with each skill
