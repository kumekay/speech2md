"""Speaker diarization overlay for a transcript.

Runs pyannote.audio `speaker-diarization-community-1` on the source audio,
then maps speaker labels onto word-level timestamps produced by
`align-transcription`. Emits:

  - <name>.speakers.md  — prose markdown grouped under `## SPEAKER_XX`
  - <name>.speakers.srt — sentence-ish SRT cues prefixed with the speaker

Why word-level input: the chunk JSON from `speech2md` is ~4 minutes per
chunk, far too coarse to attribute to a single speaker. We need the
words.json from the aligner to draw speaker boundaries at real turn
granularity.

Usage:
    speech2md --json audio.m4a
    align-transcription audio.json
    diarize-transcription audio.words.json

Needs a Hugging Face token (model gated behind user conditions): pass
via --token or the HF_TOKEN env var.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


# Sentence-ending punctuation (Latin + CJK + ellipsis), possibly followed
# by closing quotes/brackets. Matched against the tail of the running
# segment to decide whether to close it before appending the next word.
SENTENCE_END_RE = re.compile(r"[.!?。！？…]+[\"'’”)\]]*\s*$")


def _to_wav(src: Path, dst: Path) -> None:
    """Normalize source audio to 16 kHz mono wav. Matches what
    transcribe/align feed to their models and avoids format surprises
    (m4a/mp4/mp3) in pyannote's audio loader."""
    subprocess.check_call(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(src), "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", str(dst)]
    )


def fmt_srt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ------------------------------ core merge ---------------------------------


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str


def speaker_at(t: float, turns: list[Turn]) -> str | None:
    """Speaker whose turn covers time `t`, or None if `t` sits in a gap."""
    for turn in turns:
        if turn.start <= t <= turn.end:
            return turn.speaker
    return None


def nearest_speaker(t: float, turns: list[Turn]) -> str | None:
    """Closest-turn fallback for words sitting in a diarization gap."""
    if not turns:
        return None
    best = min(turns, key=lambda x: min(abs(t - x.start), abs(t - x.end)))
    return best.speaker


def group_words(words: list[Word], turns: list[Turn],
                max_gap: float) -> list[Segment]:
    """Walk words in time order and glue them into Segments. Start a new
    segment whenever the speaker changes, a pause longer than `max_gap`
    appears, or the current segment ends with sentence-closing punctuation."""
    segs: list[Segment] = []
    for w in words:
        if not (w.text or "").strip():
            continue
        mid = (w.start + w.end) / 2
        spk = speaker_at(mid, turns) or nearest_speaker(mid, turns) or "UNKNOWN"
        if not segs:
            segs.append(Segment(spk, w.start, w.end, w.text.strip()))
            continue
        last = segs[-1]
        gap = w.start - last.end
        sentence_closed = bool(SENTENCE_END_RE.search(last.text))
        if spk != last.speaker or gap > max_gap or sentence_closed:
            segs.append(Segment(spk, w.start, w.end, w.text.strip()))
        else:
            last.text = (last.text + " " + w.text.strip()).strip()
            last.end = w.end
    return segs


# ------------------------------ writers ------------------------------------


def write_markdown(segs: list[Segment], out: Path) -> None:
    """One heading per speaker-run. Consecutive segments with the same
    speaker collapse into a single paragraph; the heading only repeats
    when the speaker actually changes."""
    parts: list[str] = []
    current: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if current is not None and buf:
            parts.append(f"## {current}\n\n{' '.join(buf).strip()}\n")

    for seg in segs:
        if seg.speaker != current:
            flush()
            current = seg.speaker
            buf = [seg.text.strip()]
        else:
            buf.append(seg.text.strip())
    flush()
    out.write_text("\n".join(parts), encoding="utf-8")


def write_srt(segs: list[Segment], out: Path) -> None:
    with out.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segs, 1):
            f.write(f"{i}\n")
            f.write(f"{fmt_srt_ts(seg.start)} --> {fmt_srt_ts(seg.end)}\n")
            f.write(f"{seg.speaker}: {seg.text.strip()}\n\n")


# -------------------------- pyannote integration ---------------------------


def turns_from_pipeline_output(output) -> list[Turn]:
    """Pull `(start, end, speaker)` triples out of a pyannote pipeline
    result. Prefer `exclusive_speaker_diarization` (non-overlapping, so
    the word→speaker mapping is unambiguous) and fall back to the
    standard `speaker_diarization` if the pipeline variant doesn't
    expose the exclusive view."""
    diarization = (
        getattr(output, "exclusive_speaker_diarization", None)
        or getattr(output, "speaker_diarization", None)
        or output  # older pyannote returns an Annotation directly
    )
    turns: list[Turn] = []
    for segment, _track, speaker in diarization.itertracks(yield_label=True):
        turns.append(Turn(float(segment.start), float(segment.end), str(speaker)))
    turns.sort(key=lambda t: t.start)
    return turns


# ---------------------------------- CLI ------------------------------------


def _diarize_one(pipeline, torch, words_json: Path, *, pipeline_kwargs: dict,
                 max_gap: float, out_md: Path | None,
                 out_srt: Path | None, audio_override: Path | None) -> int:
    """Run diarization for a single words JSON. Returns 0 on success."""
    data = json.loads(words_json.read_text(encoding="utf-8"))
    words_raw = data.get("words")
    if not words_raw:
        print(f"[{words_json.name}] no 'words' key — run align-transcription first.",
              file=sys.stderr)
        return 2

    audio_path = audio_override or Path(data["source"])
    if not audio_path.exists():
        print(f"[{words_json.name}] audio not found: {audio_path}", file=sys.stderr)
        return 2

    workdir = Path(tempfile.mkdtemp(prefix="speech2md-diarize-"))
    try:
        wav = workdir / "audio.wav"
        _to_wav(audio_path, wav)

        # Load the wav ourselves and pass a waveform tensor rather than a
        # path. Modern torchaudio/pyannote routes path inputs through
        # torchcodec, whose wheel ABI is tightly coupled to a specific
        # torch version — combined with vLLM's torch pin it's a constant
        # source of version hell. Bypassing that entirely with soundfile
        # keeps the diarize step robust across torch/ffmpeg upgrades.
        import soundfile as sf
        from speech2md._gpu import silenced_stderr
        samples, sr = sf.read(str(wav), dtype="float32")
        waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, T) mono

        print(f"[{words_json.name}] running diarization ...")
        with silenced_stderr():
            output = pipeline(
                {"waveform": waveform, "sample_rate": sr},
                **pipeline_kwargs,
            )
        turns = turns_from_pipeline_output(output)
        speakers = {t.speaker for t in turns}
        print(f"  {len(turns)} turns across {len(speakers)} speakers")

        words = [
            Word(float(w["start"]), float(w["end"]), str(w.get("text") or ""))
            for w in words_raw
        ]
        segs = group_words(words, turns, max_gap)
        print(f"  {len(segs)} speaker segments")

        base_stem = words_json.stem
        if base_stem.endswith(".words"):
            base_stem = base_stem[: -len(".words")]
        md = out_md or words_json.with_name(base_stem + ".speakers.md")
        srt = out_srt or words_json.with_name(base_stem + ".speakers.srt")

        write_markdown(segs, md)
        write_srt(segs, srt)
        print(f"  wrote {md.name} and {srt.name}")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("words_json", type=Path, nargs="+",
                   help="words JSON(s) from align-transcription. Multiple "
                        "inputs share a single pipeline load.")
    p.add_argument("--audio", type=Path,
                   help="override source audio path (single-input only)")
    p.add_argument("--pipeline", default="pyannote/speaker-diarization-community-1",
                   help="pyannote pipeline name (default: community-1)")
    p.add_argument("--token",
                   default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
                   help="Hugging Face token (default: $HF_TOKEN / $HUGGINGFACE_TOKEN)")
    p.add_argument("--num-speakers", type=int, default=None,
                   help="fix the speaker count (skip auto-detect)")
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--max-gap", type=float, default=1.0,
                   help="split an SRT cue when the pause between words "
                        "exceeds this (seconds, default 1.0)")
    p.add_argument("--out-md", type=Path,
                   help="markdown output path (single-input only; "
                        "default: <input>.speakers.md)")
    p.add_argument("--out-srt", type=Path,
                   help="SRT output path (single-input only; "
                        "default: <input>.speakers.srt)")
    args = p.parse_args()

    if len(args.words_json) > 1 and (args.audio or args.out_md or args.out_srt):
        print("error: --audio / --out-md / --out-srt are only valid with a "
              "single input JSON", file=sys.stderr)
        return 2

    from speech2md._gpu import require_pyannote, require_torch, silenced_stderr
    torch = require_torch()
    # pyannote imports torchcodec at module import; when the torchcodec
    # wheel ABI doesn't match the installed torch, a long traceback-shaped
    # warning prints to stderr. We feed pyannote waveforms (not paths), so
    # torchcodec is never actually needed — hide the noise.
    with silenced_stderr():
        Pipeline = require_pyannote()

    # Token resolution: explicit arg/env wins. Otherwise pass `True` so
    # huggingface_hub falls back to whatever `huggingface-cli login` left
    # in ~/.cache/huggingface/token. Accept the community-1 terms on the
    # model page first, or this call will 401.
    token_arg = args.token if args.token else True

    print(f"loading {args.pipeline} ...")
    with silenced_stderr():
        pipeline = Pipeline.from_pretrained(args.pipeline, token=token_arg)
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

    pipeline_kwargs: dict = {}
    if args.num_speakers is not None:
        pipeline_kwargs["num_speakers"] = args.num_speakers
    else:
        if args.min_speakers is not None:
            pipeline_kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers is not None:
            pipeline_kwargs["max_speakers"] = args.max_speakers

    failures: list[tuple[Path, str]] = []
    for wj in args.words_json:
        try:
            if _diarize_one(
                pipeline, torch, wj,
                pipeline_kwargs=pipeline_kwargs,
                max_gap=args.max_gap,
                out_md=args.out_md,
                out_srt=args.out_srt,
                audio_override=args.audio,
            ) != 0:
                failures.append((wj, "missing words/audio"))
        except Exception as e:  # noqa: BLE001
            print(f"[{wj.name}] FAILED: {e}", file=sys.stderr)
            failures.append((wj, str(e)))

    rc = 1 if failures else 0
    if failures:
        print(f"\n{len(failures)} failures:", file=sys.stderr)
        for wj, err in failures:
            print(f"  {wj}: {err}", file=sys.stderr)

    from speech2md._gpu import silence_teardown
    silence_teardown()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
