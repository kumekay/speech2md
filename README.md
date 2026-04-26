# speech2md

Transcribe long audio recordings into clean, readable prose using
[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) served through
[vLLM](https://github.com/vllm-project/vllm). Optional word-level
timestamps via the matching forced aligner.

Built around a single-GPU 24 GB workstation (RTX 3090 / 4090 class).
On that hardware it transcribes at roughly **150–250× realtime** — a
51-minute recording finishes in about 16 seconds after the model is
loaded.

## Requirements

- Python 3.11 or 3.12 (vLLM does not yet support 3.13+).
- Linux, NVIDIA GPU with ~16 GB VRAM free for transcription, +2 GB
  for alignment (or run them as separate processes on 24 GB cards).
- CUDA drivers installed and working with PyTorch.
- `ffmpeg` and `ffprobe` on `$PATH`.
- [`uv`](https://docs.astral.sh/uv/).

## Install

```bash
# from a clone (editable):
uv tool install --python 3.12 -e '/path/to/speech2md[gpu]' --force

# from git:
uv tool install --python 3.12 'speech2md[gpu] @ git+https://github.com/kumekay/speech2md'
```

Exposes three commands on `$PATH`:

- `speech2md` — audio → prose markdown.
- `align-transcription` — word-level timestamps.
- `diarize-transcription` — speaker diarization on top of the word-level
  output (who spoke when).

The `[gpu]` extra pulls `torch` + `qwen-asr[vllm]` + `pyannote.audio` and
is required at runtime.

`--force` re-installs if you already have `speech2md` as a tool without
the extra.

## Quick start

Transcribe a single file:

```bash
speech2md audio/recording.m4a
```

This writes `audio/recording.md` — one long line of prose, nothing
else.

Transcribe a whole folder in one pass (the model loads once and is
reused across files):

```bash
speech2md /path/to/recordings/*.m4a /path/to/recordings/*.mp3
```

Use `--skip-existing` to safely re-run over a folder without
re-transcribing files that already have a `.md` next to them.

## Output formats

`speech2md` can run just the transcription step, or drive the full
pipeline (transcribe → align → diarize) in a single command. Output
is controlled by four orthogonal flags:

| Flag | Effect |
|------|--------|
| `--diarize` | run speaker diarization; the `.md` becomes speaker-grouped |
| `--srt` | also emit a subtitle file — word-level without `--diarize`, sentence-ish with speaker labels when `--diarize` |
| `--json` | keep the per-chunk `.json` sidecar (and `.words.json` when alignment ran) |
| `--no-md` | suppress the `.md` output (requires `--srt` or `--json`) |

Intermediate artifacts that the pipeline produces on the way (the
chunks JSON, the word-level JSON, the pre-diarization SRT) are cleaned
up by default; pass `--json` to keep them.

The final files on disk after one of the common recipes:

```bash
speech2md audio.m4a                      # audio.md (prose)
speech2md audio.m4a --diarize            # audio.md (diarized)
speech2md audio.m4a --diarize --srt      # audio.md (diarized) + audio.srt (speaker-labeled)
speech2md audio.m4a --srt                # audio.md (prose) + audio.srt (word-level)
speech2md audio.m4a --no-md --srt --diarize    # audio.srt only (speaker-labeled)
speech2md audio.m4a --no-md --json --diarize   # audio.json + audio.words.json only
```

`align-transcription` and `diarize-transcription` are also exposed
as standalone commands for scripts that want to run a single stage
against pre-existing artifacts. Both accept multiple JSON inputs in
one call so the aligner / pyannote pipeline loads once per batch:

```bash
align-transcription a.json b.json c.json
diarize-transcription a.words.json b.words.json
```

## How it works

1. `ffprobe` reads the audio duration.
2. `ffmpeg silencedetect` finds pauses (default: ≥0.4 s at ≤-35 dB).
3. A small planner (`plan_splits` in `speech2md.transcribe`) walks the
   timeline and picks cut points inside silences closest to the
   target chunk length. If no silence is available inside the window,
   it falls back to a hard cut at `--max-chunk`. This avoids
   mid-word cuts, which were the single biggest source of chunk-edge
   garbage in the first implementation.
4. `ffmpeg -ss -to` extracts each chunk as mono 16 kHz `pcm_s16le`.
5. All chunks from the current file are fed to `vLLM` in one
   `model.transcribe([...])` call. vLLM handles continuous batching
   internally.
6. Per-chunk text is joined with a space and written as prose.

The default ASR chunk target is 15 minutes (`--target 900`) with a 20
minute hard cap (`--max-chunk 1200`). When the full pipeline runs
(`--srt` and/or `--diarize`), `speech2md` does a second ASR prep pass
just for alignment with much shorter chunks: target 75 s
(`--align-target 75`), hard cap 90 s (`--align-max-chunk 90`). That
keeps the forced aligner away from the long-span timestamp collapse
seen on multi-minute chunks.

## Word-level timestamps

With `speech2md --srt` the aligner runs as a second subprocess (vLLM
and the forced aligner together OOM on 24 GB, so they need separate
processes). In pipeline mode, `speech2md` first writes the prose `.md`
from larger ASR chunks, then runs a second short-chunk ASR prep pass to
produce the JSON consumed by the aligner. The `audio.srt` output has one
cue per word.

The aligner is `Qwen/Qwen3-ForcedAligner-0.6B` — about 1–2 GB VRAM on
its own.

## Speaker diarization

`speech2md --diarize` runs `pyannote/speaker-diarization-community-1`
over the audio and attaches speaker labels to the word-level
transcript. Under the hood that's three subprocesses chained: the
aligner (and therefore the chunks sidecar) is a prerequisite because
the 4-minute chunks are too coarse to attribute to a single speaker —
we need word-level timing to draw speaker boundaries.

Output shape:

- `.md` is grouped under `## SPEAKER_XX` headings, with consecutive
  same-speaker segments collapsed into a single paragraph.
- `.srt` (when combined with `--srt`) has one cue per contiguous
  speaker run, further split on long pauses or sentence-closing
  punctuation so cues stay subtitle-sized. Each cue is prefixed with
  the speaker label.

First-time setup for diarization:

1. Accept the user conditions on
   https://huggingface.co/pyannote/speaker-diarization-community-1.
2. `huggingface-cli login` once (or pass `--hf-token` to `speech2md` /
   set `$HF_TOKEN`). The command also picks up a cached token at
   `~/.cache/huggingface/token`.

Words are assigned to speakers by maximum overlap between the word's
aligned time span and diarization turns, with midpoint/nearest-turn
fallbacks only for exact ties or gaps. A small post-pass can shift a
single short boundary word across the speaker change when the nearby
pause structure suggests pyannote was one word late/early.

Segment boundaries are drawn on speaker change, on a pause longer than
`--max-gap` (default 1.0 s), or on sentence-closing punctuation at the
end of the current segment — whichever comes first.

## Common flags

`speech2md` output selectors (see **Output formats** above for the
full matrix):

- `--diarize` / `--srt` / `--json` / `--no-md` — pick what lands on
  disk.

`speech2md` diarization tuning (only used with `--diarize`):

- `--num-speakers N` — pin the speaker count.
- `--min-speakers` / `--max-speakers` — bound the auto-detect.
- `--max-gap 1.0` — start a new SRT cue when the pause between words
  exceeds this (seconds).
- `--hf-token` — override `$HF_TOKEN` / cached login.

`speech2md` ASR tuning:

- `--language "Russian"` — force a language instead of auto-detect.
  Useful for short clips where auto-detect sometimes guesses wrong.
- `--target 240 --max-chunk 290` — tune chunk length. Longer chunks
  mean fewer boundaries but more VRAM; shorter chunks finish sooner
  on small clips but lose cross-sentence context at the boundary.
- `--noise-db -35 --min-sil 0.4` — silence detection thresholds. Bump
  `--noise-db` towards -50 for very quiet recordings, or up towards
  -25 for noisy ones.
- `--gpu-util 0.6` — vLLM's `gpu_memory_utilization`. Raise if you
  have memory headroom and want a larger KV cache, lower if you're
  sharing the GPU with something else.
- `--batch 32` — vLLM's `max_inference_batch_size`. Rarely worth
  changing unless you've got a lot of short chunks.
- `--skip-existing` — no-op on files whose `.md` already exists.
- `--target 900` / `--max-chunk 1200` — ASR chunk planner target / hard cap.
- `--align-target 75` / `--align-max-chunk 90` — shorter chunk planner used only for the alignment-prep ASR pass in pipeline mode (`--srt` / `--diarize`).
- `--keep-chunks` — keep the temporary `wav` chunks for debugging.

`align-transcription`:

- `--aligner Qwen/Qwen3-ForcedAligner-0.6B` — swap the aligner model.
- `-o / --srt` — override output paths; single-input mode only.

`diarize-transcription`:

- `--pipeline pyannote/speaker-diarization-community-1` — swap the
  pyannote pipeline.
- `--token` / `$HF_TOKEN` / `$HUGGINGFACE_TOKEN` — Hugging Face token.
  Falls back to `~/.cache/huggingface/token` written by
  `huggingface-cli login`.
- `--num-speakers` — fix the speaker count when you know it; otherwise
  use `--min-speakers` / `--max-speakers` to bound the auto-detect.
- `--max-gap 1.0` — start a new SRT cue when the pause between words
  exceeds this (seconds).
- `--audio` — source audio path override (default: the `source` field
  in the words JSON).
- `--out-md` / `--out-srt` — override output paths.

## Repository layout

```
speech2md/
  transcribe.py      ASR pipeline (vLLM + Qwen3-ASR)
  align.py           word-level timestamps post-pass
  diarize.py         speaker diarization post-pass (pyannote.audio)
  _gpu.py            GPU-stack import guard
tests/               pytest suite (TDD for production code; see CLAUDE.md)
```

## Performance notes

Measured on an RTX 3090 (24 GB), single file:

| Audio | Chunks | Transcribe | ×realtime |
|-------|--------|------------|-----------|
| 16 min | 4      | 11.0 s     | 88×       |
| 51 min | 13     | 16.4 s     | 186×      |

Alignment of the 51-minute recording took about 50 seconds and
produced 2,263 word timestamps.

Model load itself takes ~30 seconds, so a whole folder finishes much
faster than N single-file invocations would.
