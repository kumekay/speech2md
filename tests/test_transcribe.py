"""Unit tests for the transcribe core (no GPU, no ffmpeg)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from speech2md import transcribe


@dataclass
class FakeResult:
    text: str
    language: str | None = None


class FakeModel:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def transcribe(self, audio, language=None):
        self.calls.append({"audio": list(audio), "language": language})
        return self.results


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    """Stub probe_duration / detect_silences / cut_chunk — keep defaults unless
    test overrides them."""
    monkeypatch.setattr(transcribe, "probe_duration", lambda p: 120.0)
    monkeypatch.setattr(transcribe, "detect_silences", lambda p, db, sil: [])
    monkeypatch.setattr(
        transcribe,
        "cut_chunk",
        lambda src, s, e, out: Path(out).write_bytes(b""),
    )
    return monkeypatch


def _src(tmp_path: Path) -> Path:
    src = tmp_path / "video.mp4"
    src.write_bytes(b"fake")
    return src


def test_transcribe_to_paths_writes_md_and_json(tmp_path: Path, fake_ffmpeg):
    src = _src(tmp_path)
    out_md = tmp_path / "transcript.md"
    out_json = tmp_path / "transcript.json"

    model = FakeModel([FakeResult(text="hello world", language="English")])

    data = transcribe.transcribe_to_paths(
        model, src, out_md, out_json, transcribe.TranscribeConfig()
    )

    assert out_md.read_text() == "hello world\n"
    payload = json.loads(out_json.read_text())
    assert payload == data
    assert payload["duration"] == 120.0
    assert payload["language"] == "English"
    assert payload["model"] == "Qwen/Qwen3-ASR-1.7B"
    assert len(payload["chunks"]) == 1
    assert payload["chunks"][0] == {
        "start": 0.0,
        "end": 120.0,
        "language": "English",
        "text": "hello world",
    }


def test_transcribe_to_paths_skips_json_when_none(tmp_path: Path, fake_ffmpeg):
    src = _src(tmp_path)
    out_md = tmp_path / "transcript.md"
    out_json = tmp_path / "transcript.json"

    transcribe.transcribe_to_paths(
        FakeModel([FakeResult(text="hi")]),
        src,
        out_md,
        None,
        transcribe.TranscribeConfig(),
    )

    assert out_md.read_text() == "hi\n"
    assert not out_json.exists()


def test_transcribe_to_paths_language_forwarded(tmp_path: Path, fake_ffmpeg):
    src = _src(tmp_path)
    out_md = tmp_path / "transcript.md"

    model = FakeModel([FakeResult(text="привет", language="Russian")])
    transcribe.transcribe_to_paths(
        model, src, out_md, None, transcribe.TranscribeConfig(language="Russian")
    )

    assert model.calls[0]["language"] == ["Russian"]


def test_transcribe_to_paths_language_none_when_auto(tmp_path: Path, fake_ffmpeg):
    src = _src(tmp_path)
    out_md = tmp_path / "transcript.md"

    model = FakeModel([FakeResult(text="hi")])
    transcribe.transcribe_to_paths(
        model, src, out_md, None, transcribe.TranscribeConfig()  # language=None
    )

    assert model.calls[0]["language"] is None


def test_transcribe_to_paths_multi_chunk(tmp_path: Path, monkeypatch, fake_ffmpeg):
    # 600s duration, no silences → plan_splits cuts at 240 and 480 → 3 chunks.
    monkeypatch.setattr(transcribe, "probe_duration", lambda p: 600.0)

    src = _src(tmp_path)
    out_md = tmp_path / "transcript.md"

    model = FakeModel(
        [
            FakeResult(text="first"),
            FakeResult(text="second"),
            FakeResult(text="third"),
        ]
    )
    transcribe.transcribe_to_paths(
        model, src, out_md, None, transcribe.TranscribeConfig()
    )

    assert out_md.read_text() == "first second third\n"
    assert len(model.calls[0]["audio"]) == 3


def test_transcribe_to_paths_skips_empty_text(tmp_path: Path, fake_ffmpeg, monkeypatch):
    monkeypatch.setattr(transcribe, "probe_duration", lambda p: 600.0)

    src = _src(tmp_path)
    out_md = tmp_path / "transcript.md"
    out_json = tmp_path / "transcript.json"

    model = FakeModel(
        [
            FakeResult(text="one"),
            FakeResult(text=""),  # empty chunk (e.g., long silence)
            FakeResult(text="three"),
        ]
    )
    transcribe.transcribe_to_paths(
        model, src, out_md, out_json, transcribe.TranscribeConfig()
    )

    assert out_md.read_text() == "one three\n"
    payload = json.loads(out_json.read_text())
    # JSON sidecar keeps the empty chunk (caller / aligner may want it).
    assert len(payload["chunks"]) == 3
