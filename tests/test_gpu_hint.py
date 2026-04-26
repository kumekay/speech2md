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


def test_require_cuda_no_devices_prints_clear_error(capsys):
    from speech2md import _gpu

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

    class FakeTorch:
        cuda = FakeCuda()

    with pytest.raises(SystemExit) as ei:
        _gpu.require_cuda(FakeTorch(), command="speech2md")
    assert ei.value.code == 3
    err = capsys.readouterr().err
    assert "no cuda devices" in err.lower()
    assert "speech2md" in err
    assert "nvidia-smi" in err
