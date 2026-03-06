# Semantic Reflow & Speaker Diarization

Two new pipeline stages that sit between transcription and translation:

```
run_faster_whisper.py  -->  reflow_words.py  -->  diarize_gemini.py  -->  translate_vtt.py
    (ASR + words.json)      (pause-aligned      (speaker labels via       (translation with
                              VTT cues)           Gemini audio)            speaker context)
```

## `scripts/reflow_words.py`

Re-segments word-level timestamps into subtitle cues aligned to natural speech pauses, replacing the mechanical 10-second split.

### How it works

1. Flattens all words across all VAD segments into one chronological stream
2. Scans for gaps between consecutive words where `gap >= pause_threshold` (default 300ms)
3. Groups words between gaps into cues
4. Applies duration constraints:
   - Groups exceeding `max_cue_s` split at the largest internal gap
   - If no meaningful gap exists, falls back to greedy word-boundary split
   - Groups shorter than `min_cue_s` merge into the next group (prevents isolated particles)
5. Reconstructs timing from first/last word in each group

### CLI (standalone)

```bash
python scripts/reflow_words.py \
  --input episode/transcription/406_faster_v2_words.json \
  --output episode/transcription/406_reflow.vtt \
  --pause-ms 300 \
  --max-cue-s 10 \
  --min-cue-s 0.3 \
  --stats
```

| Flag | Default | Description |
|---|---|---|
| `--input` | (required) | Word timestamps JSON from faster-whisper |
| `--output` | `<input_stem>_reflow.vtt` | Output VTT path |
| `--pause-ms` | `300` | Gap threshold (ms) to trigger a cue break |
| `--max-cue-s` | `10` | Maximum cue duration before forced split |
| `--min-cue-s` | `0.3` | Minimum cue duration; shorter cues merge forward |
| `--stats` | off | Print duration min/avg/max statistics |

### Integration with `run_faster_whisper.py`

```bash
# New: semantic reflow
python scripts/run_faster_whisper.py --reflow --reflow-pause-ms 300

# Existing: mechanical split (unchanged)
python scripts/run_faster_whisper.py --max-cue-s 10
```

When `--reflow` is passed, the pipeline calls `reflow_words()` instead of `split_long_segments()`. Both paths coexist. The `--max-cue-s` value is still respected as the upper bound during reflow.

### Importable API

```python
from reflow_words import reflow_words

cues = reflow_words(
    segments,              # list[dict] from words.json
    pause_threshold=0.3,   # seconds
    max_cue_s=10.0,
    min_cue_s=0.3,
)
# Returns list[dict] with keys: start, end, text, words
```

---

## `scripts/diarize_gemini.py`

Annotates an existing VTT transcript with speaker labels using Gemini's multimodal audio understanding. No local ML dependencies — just the `google-genai` SDK and ffmpeg.

### How it works

1. Parses the input VTT transcript
2. Chunks cues into time-based groups (default 5 minutes)
3. For each chunk:
   - Extracts a mono MP3 audio slice via ffmpeg (64kbps, ~1MB/min)
   - Sends audio bytes inline (`Part.from_bytes`) + transcript cues to Gemini
   - Gemini returns speaker labels per cue + a speaker identity map
4. Speaker map accumulates across chunks for consistency
5. Outputs a diarized VTT with `Speaker: text` prefixes

### CLI

```bash
python scripts/diarize_gemini.py \
  --video episode/source/video.mp4 \
  --transcript episode/transcription/406_reflow.vtt \
  --glossary episode/glossary/translation_glossary_v2.tsv \
  --output episode/transcription/406_reflow_diarized.vtt \
  --model gemini-2.5-pro \
  --chunk-minutes 5
```

| Flag | Default | Description |
|---|---|---|
| `--video` | (required) | Source video file (audio extracted via ffmpeg) |
| `--transcript` | (required) | Input VTT transcript to annotate |
| `--output` | `<transcript_stem>_diarized.vtt` | Output diarized VTT |
| `--glossary` | (none) | Glossary TSV — talent names help speaker identification |
| `--model` | `gemini-2.5-pro` / `$GEMINI_MODEL` | Gemini model to use |
| `--chunk-minutes` | `5` | Audio chunk duration in minutes |
| `--speaker-map` | (none) | Additional output path for speaker map JSON |

### Output format

VTT cues get speaker prefixes:

```
00:03.600 --> 00:04.580
フジモン: みなさんよろしくお願いします!

00:05.780 --> 00:07.020
千原ジュニア: さあ、それでは参りましょう!
```

A `.speakers.json` file is written alongside the output VTT:

```json
{
  "Speaker A": "フジモン",
  "Speaker B": "千原ジュニア"
}
```

### Audio size budget

A 45-minute episode at mono 64kbps MP3 is ~20MB, well within Vertex's 100MB inline limit. Each 5-minute chunk is ~2.4MB.

---

## Translation with speaker labels

`translate_vtt.py` automatically detects speaker-labeled cues (the `Name: text` pattern). When >20% of cues have labels, it adds an instruction to the system prompt:

> Speaker labels are provided. Preserve them in the output as-is. Maintain consistent voice characterization per speaker.

No flags needed — detection is automatic. Non-diarized VTTs translate identically to before.

---

## Typical workflow

```bash
# 1. Transcribe with reflow
python scripts/run_faster_whisper.py --reflow

# 2. Diarize (optional — useful for panel/commentary segments)
python scripts/diarize_gemini.py \
  --video samples/episodes/ep406/source/video.mp4 \
  --transcript samples/episodes/ep406/transcription/video_faster_stock.vtt \
  --glossary samples/episodes/ep406/glossary/translation_glossary_v2.tsv

# 3. Translate (speaker labels passed through automatically)
python scripts/translate_vtt.py \
  --backend vertex \
  --input samples/episodes/ep406/transcription/video_faster_stock_diarized.vtt \
  --glossary samples/episodes/ep406/glossary/translation_glossary_v2.tsv
```

## Tuning tips

- **Pause threshold**: 300ms works well for variety shows. Try 200ms for fast panel banter, 400ms for slower narration.
- **Chunk minutes**: 5 minutes balances output token limits with speaker consistency. Increase if Gemini handles full episodes reliably.
- **Glossary for diarization**: The glossary TSV's context column is scanned for keywords like "talent", "person", "cast", "comedian" to extract known names. Add these tags to help Gemini identify speakers.

---

## Alternative approach: Gemini transcription + local forced alignment

Instead of using faster-whisper for ASR, transcribe with Gemini (which gives better Japanese comprehension, speaker diarization, and glossary awareness in a single pass) and then recover word-level timestamps locally via forced alignment.

### Pipeline

```
Gemini (audio + glossary)  →  transcript + speakers (no timestamps)
         ↓
stable-ts / ctc-forced-aligner  →  word-level timestamps
         ↓
reflow_words.py            →  VTT cues at natural pauses
```

### Forced alignment tools evaluated

| Tool | VRAM | Romanization? | External transcript | Japanese quality |
|---|---|---|---|---|
| **stable-ts** | ~2-10GB (whisper model) | No | First-class `align()` API | Best — Whisper cross-attention, works directly on Japanese text |
| **ctc-forced-aligner** | ~1.2GB | Yes (internal) | Native CLI + API | Decent — kanji romanization is lossy but usable for alignment |
| **torchaudio MMS_FA** | ~1.2GB | Yes (manual) | Native API | Same engine as ctc-forced-aligner, more DIY |
| **WhisperX** | ~1.2GB (align only) | No | Hacky workaround | Mediocre — old wav2vec2 Japanese model, 20% CER |
| **MFA** | CPU, ~4-8GB RAM | No (uses G2P) | Primary use case | Good if configured, heavy setup |

### Recommended: `stable-ts`

```python
import stable_whisper
model = stable_whisper.load_model('large-v3')  # or 'medium'/'small' for less VRAM
result = model.align('audio.mp3', transcript_text, language='ja')
result.to_vtt('output.vtt')
```

- Uses Whisper's cross-attention weights (DTW-based) — no separate alignment model needed
- Works directly on Japanese characters, no romanization
- Can use smaller Whisper models (medium ~5GB, small ~2GB) since only aligning, not transcribing
- Outputs word-level timestamps to VTT/SRT/JSON

### Fallback: `ctc-forced-aligner`

```bash
ctc-forced-aligner --audio_path audio.wav --text_path transcript.txt \
    --language jpn --romanize --split_size char
```

- Only 1.2GB VRAM (MMS-300M model)
- Handles Japanese romanization internally via uroman
- Character-level alignment available
- Kanji romanization is inherently lossy but sufficient for timestamp alignment

### Status

Not yet implemented. The current pipeline (faster-whisper → reflow → diarize_gemini) is functional. This alternative would collapse it into (Gemini → align → reflow) with potentially better transcription quality.

---

## Other explored: VibeVoice-ASR

Microsoft's joint ASR + diarization + timestamps model (9B params, Qwen2-7B decoder + ConvNeXt speech encoder). Does everything in one 60-minute pass.

- 4-bit quants available (~5GB VRAM), ONNX conversion exists
- Japanese support claimed (50+ languages) but unverified on variety show audio
- RDNA 4 (gfx1201) PyTorch support is uncertain — ONNX Runtime may be more practical
- Not yet tested; worth revisiting when tooling matures
