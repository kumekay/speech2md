# speech2md project conventions

## Production code: TDD

Red/green TDD for all non-experimental code.

1. Write failing test first (red).
2. Minimum code to pass (green).
3. Refactor.

Tests ship in the same PR as the code they cover. A production PR without
tests is not reviewable.

"Production" = anything under the installed package (`speech2md/`,
entry-point commands).

## Model choices

- **Do not use** `gemma4:e4b` via ollama's default 4-bit quant. The 4-bit
  quant is too lossy for this model. Prefer vLLM or llama.cpp with a
  higher-precision quant (Q8_0 or bf16).
- ASR: `Qwen/Qwen3-ASR-1.7B` via vLLM.
- Forced aligner: `Qwen/Qwen3-ForcedAligner-0.6B`.
- Diarization: `pyannote/speaker-diarization-community-1`.

## README upkeep

Update `README.md` in the same change whenever user-facing behavior
shifts: new/renamed/removed entry-point command, new runtime
dependency in `[gpu]`, new CLI flag worth documenting, or a changed
output file. Internal refactors don't need a README touch.

## Pipeline shape

`speech2md --diarize` / `--srt` drives the full three-stage pipeline
by subprocessing itself (transcribe), then `align-transcription`, then
`diarize-transcription`. Each stage gets a clean GPU because vLLM /
torch don't reliably release VRAM without process teardown — that
constraint is why the intermediate stages are separate processes, not
in-process calls. `align-transcription` and `diarize-transcription`
accept multiple JSON inputs so each model loads once per batch.
