"""Import the GPU stack (`torch`, `qwen_asr`) with a readable error when
the `[gpu]` optional-dependency group isn't installed."""
from __future__ import annotations

import contextlib
import os
import sys
import warnings

# Quiet the GPU stack by default. Each var is `setdefault`, so users can
# override by exporting before running.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", category=SyntaxWarning)


@contextlib.contextmanager
def silenced_stderr():
    """fd-level redirect of stderr to /dev/null. Subprocesses spawned inside
    inherit the devnull fd, so their stderr stays silenced even after the
    parent restores its own."""
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)


def silence_teardown() -> None:
    """Redirect parent stderr to /dev/null so the parent's own teardown
    chatter (vLLM `Engine core ... died` ERROR, NCCL `destroy_process_group`
    warning) doesn't trail after the last useful line. Call right before
    returning from `main()`."""
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)

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


def require_pyannote():
    require_torch()
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print(
            "error: pyannote.audio is not installed.\n\n"
            "Install the [gpu] extra (or add 'pyannote.audio' manually):\n"
            "    uv tool install -e '<speech2md-path>[gpu]' --force",
            file=sys.stderr,
        )
        raise SystemExit(3)
    return Pipeline
