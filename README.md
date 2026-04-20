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

Exposes two commands on `$PATH`:

- `speech2md` — audio → prose markdown.
- `align-transcription` — word-level timestamps.

The `[gpu]` extra pulls `torch` + `qwen-asr[vllm]` and is required at
runtime.

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

By default only a prose `.md` is written. The JSON sidecar is
opt-in:

| Flag | File | Content |
|------|------|---------|
| default | `<name>.md` | flat prose, one long line, no chunk markers |
| `--json` | `<name>.json` | per-chunk start/end/text, used as input to `align-transcription` |

`align-transcription` then produces:

| File | Content |
|------|---------|
| `<name>.words.json` | `[{start, end, text}, …]` flat list of words |
| `<name>.srt` | one SRT cue per word |

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

The default chunk target is 4 minutes (`--target 240`) with a 4:50
hard cap (`--max-chunk 290`). The forced aligner's input cap is
5 minutes, so staying at or below 290 s keeps the chunks compatible
with the align step too.

## Word-level timestamps

vLLM and the forced aligner together OOM on 24 GB, so alignment runs
as a separate pass over the JSON sidecar:

```bash
speech2md --json audio/recording.m4a
align-transcription audio/recording.json
```

Outputs `audio/recording.words.json` and `audio/recording.srt`. The
aligner is `Qwen/Qwen3-ForcedAligner-0.6B` — about 1–2 GB VRAM on its
own.

## Common flags

`speech2md`:

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
- `--json` — also emit the per-chunk sidecar for `align-transcription`.
- `--skip-existing` — no-op on files whose `.md` already exists.
- `--keep-chunks` — keep the temporary `wav` chunks for debugging.

`align-transcription`:

- `--aligner Qwen/Qwen3-ForcedAligner-0.6B` — swap the aligner model.
- `-o / --srt` — override output paths; otherwise derived from input.

## Repository layout

```
speech2md/
  transcribe.py      ASR pipeline (vLLM + Qwen3-ASR)
  align.py           word-level timestamps post-pass
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
