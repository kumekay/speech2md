"""Import the GPU stack (`torch`, `qwen_asr`) with a readable error when
the `[gpu]` optional-dependency group isn't installed."""
from __future__ import annotations

import os
import sys
import warnings

# Quiet the GPU stack by default. Each var is `setdefault`, so users can
# override by exporting before running.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", category=SyntaxWarning)


def silence_teardown() -> None:
    """Redirect stderr to /dev/null so vLLM/NCCL shutdown noise (engine-core
    `died unexpectedly` ERROR, `destroy_process_group` warning) doesn't trail
    after the last useful line. Call right before returning from `main()`."""
    sys.stderr.flush()
    sys.stderr = open(os.devnull, "w")

_HINT = (
    "This command needs the GPU stack (torch + qwen-asr + vLLM). Install with:\n"
    "    uv tool install -e '<speech2md-path>[gpu]' --force\n"
    "or for a release install:\n"
    "    uv tool install 'speech2md[gpu]' --force"
)


def require_torch():
    try:
        import torch
    except ImportError:
        print(f"error: torch is not installed.\n\n{_HINT}", file=sys.stderr)
        raise SystemExit(3)
    return torch


def require_qwen_asr_llm():
    require_torch()
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError:
        print(f"error: qwen-asr is not installed.\n\n{_HINT}", file=sys.stderr)
        raise SystemExit(3)
    return Qwen3ASRModel


def require_forced_aligner():
    require_torch()
    try:
        from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner
    except ImportError:
        print(f"error: qwen-asr is not installed.\n\n{_HINT}", file=sys.stderr)
        raise SystemExit(3)
    return Qwen3ForcedAligner
