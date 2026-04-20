import sys

import pytest


def test_require_torch_missing_prints_hint(monkeypatch, capsys):
    from speech2md import _gpu

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SystemExit) as ei:
        _gpu.require_torch()
    assert ei.value.code == 3
    err = capsys.readouterr().err
    assert "torch" in err.lower()
    assert "[gpu]" in err


def test_require_qwen_asr_missing_prints_hint(monkeypatch, capsys):
    from speech2md import _gpu

    # torch is fine; qwen_asr is not.
    monkeypatch.setitem(sys.modules, "qwen_asr", None)
    with pytest.raises(SystemExit) as ei:
        _gpu.require_qwen_asr_llm()
    assert ei.value.code == 3
    err = capsys.readouterr().err
    assert "qwen-asr" in err.lower()
    assert "[gpu]" in err
