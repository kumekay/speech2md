"""Unit tests for the transcribe core (no GPU, no ffmpeg)."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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
        model,
        src,
        out_md,
        None,
        transcribe.TranscribeConfig(target=240.0, max_chunk=290.0),
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
        model,
        src,
        out_md,
        out_json,
        transcribe.TranscribeConfig(target=240.0, max_chunk=290.0),
    )

    assert out_md.read_text() == "one three\n"
    payload = json.loads(out_json.read_text())
    # JSON sidecar keeps the empty chunk (caller / aligner may want it).
    assert len(payload["chunks"]) == 3


# ---------------- pipeline output reconciliation -----------------


def _seed_pipeline_outputs(src: Path, *, with_align: bool,
                           with_diarize: bool) -> dict[str, Path]:
    """Create the files each pipeline stage would actually produce, so
    the reconciliation logic has realistic inputs to prune/rename. If
    align didn't run there's no words.json or word-SRT. If diarize
    didn't run there's no speakers.md / speakers.srt."""
    stem = src.stem
    files = {
        "prose_md": src.with_name(f"{stem}.md"),
        "chunks_json": src.with_name(f"{stem}.json"),
    }
    files["prose_md"].write_text("prose md", encoding="utf-8")
    files["chunks_json"].write_text("{}", encoding="utf-8")
    if with_align:
        files["words_json"] = src.with_name(f"{stem}.words.json")
        files["word_srt"] = src.with_name(f"{stem}.srt")
        files["words_json"].write_text("{}", encoding="utf-8")
        files["word_srt"].write_text("word srt", encoding="utf-8")
    if with_diarize:
        files["speakers_md"] = src.with_name(f"{stem}.speakers.md")
        files["speakers_srt"] = src.with_name(f"{stem}.speakers.srt")
        files["speakers_md"].write_text("diarized md", encoding="utf-8")
        files["speakers_srt"].write_text("speakers srt", encoding="utf-8")
    return files


def test_reconcile_diarize_only_keeps_diarized_md(tmp_path: Path):
    src = tmp_path / "audio.m4a"
    f = _seed_pipeline_outputs(src, with_align=True, with_diarize=True)
    transcribe._reconcile_outputs(
        src, diarize=True, srt=False, json_out=False, no_md=False
    )
    assert f["prose_md"].read_text() == "diarized md"
    assert not f["chunks_json"].exists()
    assert not f["words_json"].exists()
    assert not f["word_srt"].exists()
    assert not f["speakers_md"].exists()
    assert not f["speakers_srt"].exists()


def test_reconcile_diarize_with_srt_renames_speakers_srt(tmp_path: Path):
    src = tmp_path / "audio.m4a"
    f = _seed_pipeline_outputs(src, with_align=True, with_diarize=True)
    transcribe._reconcile_outputs(
        src, diarize=True, srt=True, json_out=False, no_md=False
    )
    assert f["prose_md"].read_text() == "diarized md"
    assert f["word_srt"].read_text() == "speakers srt"
    assert not f["speakers_md"].exists()
    assert not f["speakers_srt"].exists()
    assert not f["chunks_json"].exists()
    assert not f["words_json"].exists()


def test_reconcile_srt_only_no_diarize_keeps_word_srt(tmp_path: Path):
    src = tmp_path / "audio.m4a"
    f = _seed_pipeline_outputs(src, with_align=True, with_diarize=False)
    transcribe._reconcile_outputs(
        src, diarize=False, srt=True, json_out=False, no_md=False
    )
    assert f["prose_md"].read_text() == "prose md"
    assert f["word_srt"].read_text() == "word srt"
    assert not f["chunks_json"].exists()
    assert not f["words_json"].exists()


def test_reconcile_no_md_srt_only(tmp_path: Path):
    src = tmp_path / "audio.m4a"
    f = _seed_pipeline_outputs(src, with_align=True, with_diarize=True)
    transcribe._reconcile_outputs(
        src, diarize=True, srt=True, json_out=False, no_md=True
    )
    assert not f["prose_md"].exists()
    assert not f["speakers_md"].exists()
    assert f["word_srt"].read_text() == "speakers srt"


def test_reconcile_json_only_no_md_no_diarize(tmp_path: Path):
    src = tmp_path / "audio.m4a"
    f = _seed_pipeline_outputs(src, with_align=False, with_diarize=False)
    transcribe._reconcile_outputs(
        src, diarize=False, srt=False, json_out=True, no_md=True
    )
    assert not f["prose_md"].exists()
    assert f["chunks_json"].exists()


def test_reconcile_json_and_diarize_keeps_words_json(tmp_path: Path):
    src = tmp_path / "audio.m4a"
    f = _seed_pipeline_outputs(src, with_align=True, with_diarize=True)
    transcribe._reconcile_outputs(
        src, diarize=True, srt=False, json_out=True, no_md=True
    )
    assert not f["prose_md"].exists()
    assert not f["speakers_md"].exists()
    assert f["chunks_json"].exists()
    assert f["words_json"].exists()


def test_reconcile_default_prose_only(tmp_path: Path):
    """Fast path — only prose .md written by speech2md, no intermediates."""
    src = tmp_path / "audio.m4a"
    prose = src.with_name("audio.md")
    prose.write_text("prose", encoding="utf-8")
    transcribe._reconcile_outputs(
        src, diarize=False, srt=False, json_out=False, no_md=False
    )
    assert prose.read_text() == "prose"


def test_validate_outputs_no_md_requires_output():
    with pytest.raises(SystemExit):
        transcribe._validate_outputs(no_md=True, srt=False, json_out=False)


def test_validate_outputs_no_md_with_srt_ok():
    transcribe._validate_outputs(no_md=True, srt=True, json_out=False)


def test_validate_outputs_no_md_with_json_ok():
    transcribe._validate_outputs(no_md=True, srt=False, json_out=True)


def _pipeline_args(src: Path, **overrides) -> SimpleNamespace:
    args = SimpleNamespace(
        inputs=[src],
        target=900.0,
        max_chunk=1200.0,
        align_target=75.0,
        align_max_chunk=90.0,
        noise_db=-35,
        min_sil=0.4,
        language=None,
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_util=0.6,
        max_model_len=8192,
        batch=32,
        max_new_tokens=4096,
        keep_chunks=False,
        skip_existing=False,
        diarize=False,
        srt=True,
        json=True,
        no_md=False,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
        max_gap=None,
        hf_token=None,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def test_run_pipeline_uses_short_alignment_chunks_and_preserves_prose_md(
    tmp_path: Path, monkeypatch
):
    src = tmp_path / "audio.m4a"
    src.write_bytes(b"fake")
    args = _pipeline_args(src)
    calls: list[list[str]] = []

    def fake_call(cmd):
        calls.append(cmd)
        if cmd[:3] == [sys.executable, "-m", "speech2md.transcribe"]:
            if "--json" in cmd:
                # Phase 2 with --no-md writes the json sidecar only.
                src.with_suffix(".json").write_text("{}", encoding="utf-8")
            else:
                src.with_suffix(".md").write_text("long prose\n", encoding="utf-8")
        elif cmd[:3] == [sys.executable, "-m", "speech2md.align"]:
            src.with_suffix(".words.json").write_text("{}", encoding="utf-8")
            src.with_suffix(".srt").write_text("word srt", encoding="utf-8")
        return 0

    monkeypatch.setattr(transcribe.subprocess, "call", fake_call)

    assert transcribe._run_pipeline(args) == 0
    assert src.with_suffix(".md").read_text(encoding="utf-8") == "long prose\n"
    assert calls[0][:3] == [sys.executable, "-m", "speech2md.transcribe"]
    assert "--json" not in calls[0]
    assert "--no-md" not in calls[0]
    assert calls[0][calls[0].index("--target") + 1] == "900.0"
    assert calls[0][calls[0].index("--max-chunk") + 1] == "1200.0"
    assert calls[1][:3] == [sys.executable, "-m", "speech2md.transcribe"]
    assert "--json" in calls[1]
    assert "--no-md" in calls[1]
    assert calls[1][calls[1].index("--target") + 1] == "75.0"
    assert calls[1][calls[1].index("--max-chunk") + 1] == "90.0"
    assert calls[2][:3] == [sys.executable, "-m", "speech2md.align"]


def test_transcribe_one_skip_existing_checks_actual_outputs(
    tmp_path: Path, monkeypatch
):
    """Regression for the diarize bug: with --no-md --json --skip-existing,
    the skip check must look at the .json sidecar — not at .md, which is
    irrelevant to this invocation. Previously phase 2 of the pipeline
    short-circuited because phase 1's .md confused the skip check."""
    src = _src(tmp_path)
    out_md = src.with_suffix(".md")
    out_md.write_text("phase 1 prose", encoding="utf-8")  # left over from phase 1

    ran = []
    monkeypatch.setattr(
        transcribe, "transcribe_to_paths",
        lambda *a, **kw: ran.append((a, kw)) or {},
    )

    args = SimpleNamespace(
        inputs=[src],
        target=75.0, max_chunk=90.0, align_target=75.0, align_max_chunk=90.0,
        noise_db=-35, min_sil=0.4, language=None,
        model="Qwen/Qwen3-ASR-1.7B", gpu_util=0.6, max_model_len=8192,
        batch=32, max_new_tokens=4096, keep_chunks=False,
        skip_existing=True, no_md=True, json=True,
    )

    transcribe.transcribe_one(object(), src, args)
    assert ran, "phase 2 must run; the .md from phase 1 should not block it"


def test_transcribe_one_skip_existing_skips_when_outputs_present(
    tmp_path: Path, monkeypatch
):
    src = _src(tmp_path)
    src.with_suffix(".json").write_text("{}", encoding="utf-8")

    ran = []
    monkeypatch.setattr(
        transcribe, "transcribe_to_paths",
        lambda *a, **kw: ran.append((a, kw)) or {},
    )

    args = SimpleNamespace(
        inputs=[src],
        target=75.0, max_chunk=90.0, align_target=75.0, align_max_chunk=90.0,
        noise_db=-35, min_sil=0.4, language=None,
        model="Qwen/Qwen3-ASR-1.7B", gpu_util=0.6, max_model_len=8192,
        batch=32, max_new_tokens=4096, keep_chunks=False,
        skip_existing=True, no_md=True, json=True,
    )

    transcribe.transcribe_one(object(), src, args)
    assert not ran


def test_transcribe_to_paths_no_md_writes_only_json(tmp_path: Path, fake_ffmpeg):
    src = _src(tmp_path)
    out_md = tmp_path / "transcript.md"
    out_json = tmp_path / "transcript.json"

    transcribe.transcribe_to_paths(
        FakeModel([FakeResult(text="hi")]),
        src, None, out_json, transcribe.TranscribeConfig(),
    )

    assert not out_md.exists()
    assert out_json.exists()
