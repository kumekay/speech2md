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
