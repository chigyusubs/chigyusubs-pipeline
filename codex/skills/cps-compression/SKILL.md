---
name: cps-compression
description: Compress subtitle cues that exceed the hard CPS (characters per second) limit after translation. Use when translated subtitles have CPS overruns that need to be shortened while preserving meaning and punchlines. Codex-interactive workflow using prepare/next-cue/apply-cue/finalize.
---

# CPS Compression

Post-translation pass to compress cues that exceed the hard CPS limit. Codex reads each overrun cue with surrounding context and rewrites it to fit within the character budget.

Use `scripts/compress_cps_overruns.py` as the helper workflow:

- `prepare`
- `next-cue`
- `apply-cue`
- `status`
- `finalize`

## When To Use

After translation is complete and the diagnostics show hard CPS violations. This is a focused compression pass — not a retranslation.

## Workflow

### 1. Prepare

```
python scripts/compress_cps_overruns.py prepare \
  --input <translated_vtt> \
  --hard-cps 20
```

This scans the VTT for overruns and creates a session.

### 2. Compression Loop

**Loop until all overrun cues are compressed or a `red` review stops the session.**

1. Run `next-cue` to get the current overrun cue with context and character budget.
2. Rewrite the cue text to fit within `max_chars` while preserving meaning.
3. Write the compressed text JSON and run `apply-cue`.
4. If status is `completed` or `stopped`, exit. Otherwise, loop back to step 1.

Do NOT stop after a single cue. Do NOT wait for user confirmation between cues unless the review is `red`.

### 3. Compression Rules

For each overrun cue:

- Stay within the `max_chars` budget shown in `next-cue` output
- Keep natural, readable subtitle English
- Preserve punchlines and comedy timing
- Do not drop meaning — compress, don't delete
- Use context cues to understand what can be safely shortened
- If a cue is genuinely impossible to compress (e.g., 0.12s duration), mark review as `yellow` and move on

### 4. Finalize

```
python scripts/compress_cps_overruns.py finalize --session <path>
```

Writes the final VTT with all compressions applied.

## apply-cue JSON Format

```json
{
  "cue_id": 42,
  "text": "compressed subtitle text",
  "review": "green",
  "notes": ""
}
```

Reviews: `green` (continue), `yellow` (continue with caution), `red` (stop session).
