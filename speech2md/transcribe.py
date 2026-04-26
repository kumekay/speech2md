"""Transcribe long audio with Qwen3-ASR + vLLM.

Pipeline:
  1. `ffmpeg silencedetect` finds pauses in the audio.
  2. Plan splits land inside pauses near the target chunk length — no
     mid-word cuts.
  3. vLLM batches all chunks in one call, continuous batching on GPU.
  4. Output: flat prose markdown. Pass --json to also write a sidecar
     with per-chunk timestamps (needed by align-transcription).

Word-level timestamps (forced aligner) don't fit in the same process as
vLLM on 24 GB — run `align-transcription` as a separate step after this.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


# ------------------------------ ffmpeg helpers ------------------------------


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True,
    )
    return float(out.strip())


@dataclass
class Silence:
    start: float
    end: float

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2


def detect_silences(path: Path, noise_db: int, min_sil: float) -> list[Silence]:
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_sil}",
        "-f", "null", "-",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    starts = [float(m.group(1)) for m in re.finditer(r"silence_start:\s*(-?[\d.]+)", p.stderr)]
    ends = re.findall(r"silence_end:\s*(-?[\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)", p.stderr)
    sils: list[Silence] = []
    for s, (e, _) in zip(starts, ends):
        e = float(e)
        if e > s:
            sils.append(Silence(max(s, 0.0), e))
    if len(starts) > len(ends):
        sils.append(Silence(max(starts[-1], 0.0), probe_duration(path)))
    return sils


def plan_splits(duration: float, silences: list[Silence],
                target: float, max_chunk: float) -> list[tuple[float, float]]:
    cuts: list[float] = []
    pos = 0.0
    while duration - pos > max_chunk:
        ideal = pos + target
        limit = pos + max_chunk
        cands = [s for s in silences if pos < s.mid <= limit]
        cut = min(cands, key=lambda s: abs(s.mid - ideal)).mid if cands else min(ideal, limit)
        cuts.append(cut)
        pos = cut
    starts = [0.0] + cuts
    ends = cuts + [duration]
    return list(zip(starts, ends))


def cut_chunk(src: Path, start: float, end: float, out: Path) -> None:
    subprocess.check_call(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out)]
    )


# ------------------------------ formatting ---------------------------------


def fmt_ts(seconds: float) -> str:
    td = dt.timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    m, s = divmod(rem, 60)
    h += td.days * 24
    return f"{h:02d}:{m:02d}:{s:02d}"


# --------------------------------- config ----------------------------------


DEFAULT_ASR_TARGET = 900.0
DEFAULT_ASR_MAX_CHUNK = 1200.0
DEFAULT_ALIGN_TARGET = 75.0
DEFAULT_ALIGN_MAX_CHUNK = 90.0


@dataclass
class TranscribeConfig:
    target: float = DEFAULT_ASR_TARGET
    max_chunk: float = DEFAULT_ASR_MAX_CHUNK
    noise_db: int = -35
    min_sil: float = 0.4
    language: str | None = None
    model_name: str = "Qwen/Qwen3-ASR-1.7B"
    gpu_util: float = 0.6
    max_model_len: int = 8192
    batch: int = 32
    max_new_tokens: int = 4096
    keep_chunks: bool = False


def _config_from_args(args) -> TranscribeConfig:
    return TranscribeConfig(
        target=args.target,
        max_chunk=args.max_chunk,
        noise_db=args.noise_db,
        min_sil=args.min_sil,
        language=args.language,
        model_name=args.model,
        gpu_util=args.gpu_util,
        max_model_len=args.max_model_len,
        batch=args.batch,
        max_new_tokens=args.max_new_tokens,
        keep_chunks=args.keep_chunks,
    )


# ------------------------------ pipeline glue ------------------------------


def _validate_outputs(*, no_md: bool, srt: bool, json_out: bool) -> None:
    """`--no-md` must leave at least one output on disk."""
    if no_md and not (srt or json_out):
        print("error: --no-md requires --srt or --json (or both)", file=sys.stderr)
        raise SystemExit(2)


def _safe_unlink(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def _reconcile_outputs(src: Path, *, diarize: bool, srt: bool,
                       json_out: bool, no_md: bool) -> None:
    """Post-pipeline: rename the diarize/align intermediates into the
    user's requested final set and delete the rest. Idempotent — missing
    files are fine.

    File map (before reconcile, when the full pipeline runs):
      <stem>.md            prose md        (speech2md)
      <stem>.json          chunks sidecar  (speech2md --json)
      <stem>.words.json    word timings    (align)
      <stem>.srt           word-level SRT  (align)
      <stem>.speakers.md   diarized md     (diarize, if --diarize)
      <stem>.speakers.srt  speaker SRT     (diarize, if --diarize)
    """
    stem = src.stem
    prose_md = src.with_name(f"{stem}.md")
    chunks_json = src.with_name(f"{stem}.json")
    words_json = src.with_name(f"{stem}.words.json")
    word_srt = src.with_name(f"{stem}.srt")
    speakers_md = src.with_name(f"{stem}.speakers.md")
    speakers_srt = src.with_name(f"{stem}.speakers.srt")

    # Finalize the .md slot. When --diarize, the speakers.md replaces prose.md.
    if no_md:
        _safe_unlink(prose_md)
        _safe_unlink(speakers_md)
    elif diarize and speakers_md.exists():
        _safe_unlink(prose_md)
        speakers_md.rename(prose_md)

    # Finalize the .srt slot.
    if srt:
        if diarize and speakers_srt.exists():
            _safe_unlink(word_srt)
            speakers_srt.rename(word_srt)
        # else: word_srt (from align) is already the right file
    else:
        _safe_unlink(word_srt)
        _safe_unlink(speakers_srt)

    # Intermediate JSONs are only kept when --json is explicit.
    if not json_out:
        _safe_unlink(chunks_json)
        _safe_unlink(words_json)


# --------------------------------- core ------------------------------------


def load_model(config: TranscribeConfig):
    """Load Qwen3-ASR via vLLM. Heavy — call once, reuse across inputs.

    Spawn the engine subprocess with stderr → /dev/null so its tqdm bars
    and teardown warnings don't leak into our output. The parent's stderr
    is restored on exit so per-file failure messages still surface."""
    from speech2md._gpu import (
        require_cuda,
        require_qwen_asr_llm,
        require_torch,
        silenced_stderr,
    )
    torch = require_torch()
    require_cuda(torch, command="speech2md")
    Qwen3ASRModel = require_qwen_asr_llm()
    with silenced_stderr():
        return Qwen3ASRModel.LLM(
            model=config.model_name,
            gpu_memory_utilization=config.gpu_util,
            max_inference_batch_size=config.batch,
            max_new_tokens=config.max_new_tokens,
            max_model_len=config.max_model_len,
        )


def transcribe_to_paths(model, src: Path, out_md: Path, out_json: Path | None,
                        config: TranscribeConfig) -> dict:
    """Transcribe `src` to `out_md` (prose) and optionally `out_json`
    (per-chunk timestamps, consumed by align-transcription).
    Returns the JSON-shaped payload regardless of whether it was written."""
    duration = probe_duration(src)
    silences = detect_silences(src, config.noise_db, config.min_sil)
    segments = plan_splits(duration, silences, config.target, config.max_chunk)
    n = len(segments)
    print(f"{src.name}: duration={fmt_ts(duration)} silences={len(silences)} "
          f"plan={n} chunks")

    workdir = Path(tempfile.mkdtemp(prefix="speech2md-"))
    try:
        chunk_paths: list[Path] = []
        for i, (s, e) in enumerate(segments, 1):
            wav = workdir / f"chunk_{i:02d}.wav"
            cut_chunk(src, s, e, wav)
            chunk_paths.append(wav)

        t0 = time.time()
        lang_arg = [config.language] * n if config.language else None
        results = model.transcribe(
            audio=[str(x) for x in chunk_paths],
            language=lang_arg,
        )
        elapsed = time.time() - t0
        print(f"  transcribed in {elapsed:.1f}s ({duration / elapsed:.1f}x realtime)")

        chunks_out: list[dict] = []
        for (s, e), r in zip(segments, results):
            chunks_out.append({
                "start": round(s, 3),
                "end": round(e, 3),
                "language": getattr(r, "language", None),
                "text": (getattr(r, "text", "") or "").strip(),
            })

        prose = " ".join(c["text"] for c in chunks_out if c["text"])
        out_md.write_text(prose + "\n", encoding="utf-8")

        detected = next((c["language"] for c in chunks_out if c["language"]), None)
        data = {
            "source": str(src),
            "duration": duration,
            "model": config.model_name,
            "language": detected,
            "chunks": chunks_out,
        }

        if out_json is not None:
            out_json.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  wrote {out_md.name} and {out_json.name}")
        else:
            print(f"  wrote {out_md.name}")

        return data
    finally:
        if not config.keep_chunks:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"  chunks kept at {workdir}")


def transcribe_one(model, src: Path, args) -> None:
    """CLI shim: derive sidecar paths from `src` and call transcribe_to_paths."""
    out_md = src.with_suffix(".md")
    out_json = src.with_suffix(".json")
    if out_md.exists() and args.skip_existing:
        print(f"skip {src.name} (output exists)")
        return
    transcribe_to_paths(
        model,
        src,
        out_md,
        out_json if args.json else None,
        _config_from_args(args),
    )


def _transcribe_stage_cmd(args, *, keep_json: bool, no_md: bool,
                          target: float, max_chunk: float) -> list[str]:
    cmd = [
        sys.executable, "-m", "speech2md.transcribe",
        *(str(x) for x in args.inputs),
        "--target", str(target),
        "--max-chunk", str(max_chunk),
        "--noise-db", str(args.noise_db),
        "--min-sil", str(args.min_sil),
        "--model", args.model,
        "--gpu-util", str(args.gpu_util),
        "--max-model-len", str(args.max_model_len),
        "--batch", str(args.batch),
        "--max-new-tokens", str(args.max_new_tokens),
    ]
    if keep_json:
        cmd.append("--json")
    if no_md:
        cmd.append("--no-md")
    if args.language:
        cmd += ["--language", args.language]
    if args.keep_chunks:
        cmd.append("--keep-chunks")
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd


def _snapshot_markdown(inputs: list[Path]) -> dict[Path, str]:
    saved: dict[Path, str] = {}
    for src in inputs:
        md = src.with_name(f"{src.stem}.md")
        if md.exists():
            saved[md] = md.read_text(encoding="utf-8")
    return saved


def _restore_markdown(saved: dict[Path, str]) -> None:
    for path, content in saved.items():
        path.write_text(content, encoding="utf-8")


# ---------------------------------- main -----------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, nargs="+", help="audio file(s)")
    p.add_argument("--target", type=float, default=DEFAULT_ASR_TARGET,
                   help="ASR target chunk seconds (default 900 = 15 min)")
    p.add_argument("--max-chunk", type=float, default=DEFAULT_ASR_MAX_CHUNK,
                   help="ASR hard max chunk seconds (default 1200 = 20 min)")
    p.add_argument("--align-target", type=float, default=DEFAULT_ALIGN_TARGET,
                   help="pipeline-only: target chunk seconds for the alignment-prep ASR pass (default 75)")
    p.add_argument("--align-max-chunk", type=float, default=DEFAULT_ALIGN_MAX_CHUNK,
                   help="pipeline-only: hard max chunk seconds for the alignment-prep ASR pass (default 90)")
    p.add_argument("--noise-db", type=int, default=-35,
                   help="silencedetect noise threshold in dB (default -35)")
    p.add_argument("--min-sil", type=float, default=0.4,
                   help="silencedetect minimum silence duration in seconds (default 0.4)")
    p.add_argument("--language", default=None,
                   help="force language (e.g. 'Russian'); default auto-detect")
    p.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--gpu-util", type=float, default=0.6,
                   help="vLLM gpu_memory_utilization (default 0.6)")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--batch", type=int, default=32,
                   help="vLLM max_inference_batch_size (default 32)")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--keep-chunks", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip files whose .md already exists")
    p.add_argument("--json", action="store_true",
                   help="keep the per-chunk JSON sidecar (and the "
                        "word-level JSON when --srt/--diarize ran alignment)")
    # Output selectors
    p.add_argument("--diarize", action="store_true",
                   help="run speaker diarization; the .md is then speaker-grouped")
    p.add_argument("--srt", action="store_true",
                   help="also emit a subtitle file (word-level without --diarize, "
                        "sentence-ish with speaker labels when --diarize)")
    p.add_argument("--no-md", action="store_true",
                   help="suppress the .md output (requires --srt or --json)")
    # Diarization pass-through flags (only used when --diarize)
    p.add_argument("--num-speakers", type=int, default=None,
                   help="pin speaker count (diarize only)")
    p.add_argument("--min-speakers", type=int, default=None,
                   help="minimum speaker count (diarize only)")
    p.add_argument("--max-speakers", type=int, default=None,
                   help="maximum speaker count (diarize only)")
    p.add_argument("--max-gap", type=float, default=None,
                   help="new SRT cue when pause between words exceeds this "
                        "(diarize only, default 1.0)")
    p.add_argument("--hf-token", default=None,
                   help="Hugging Face token for the gated pyannote pipeline "
                        "(defaults to $HF_TOKEN / $HUGGINGFACE_TOKEN / cached login)")
    args = p.parse_args()

    _validate_outputs(no_md=args.no_md, srt=args.srt, json_out=args.json)

    missing = [x for x in args.inputs if not x.exists()]
    if missing:
        for x in missing:
            print(f"input not found: {x}", file=sys.stderr)
        return 2

    if args.diarize or args.srt:
        return _run_pipeline(args)
    return _run_transcribe_only(args)


def _run_transcribe_only(args) -> int:
    """Fast path: single-stage transcribe, no align/diarize subprocesses."""
    config = _config_from_args(args)
    print(f"loading vLLM {config.model_name} ...")
    t0 = time.time()
    model = load_model(config)
    print(f"loaded in {time.time() - t0:.1f}s")

    failures: list[tuple[Path, str]] = []
    for src in args.inputs:
        try:
            transcribe_one(model, src, args)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {src.name}: {e}", file=sys.stderr)
            failures.append((src, str(e)))

    rc = 1 if failures else 0
    if failures:
        print(f"\n{len(failures)} failures:", file=sys.stderr)
        for src, err in failures:
            print(f"  {src}: {err}", file=sys.stderr)

    if args.no_md or not args.json:
        # `speech2md` default-writes .md and optionally .json; reconcile
        # to honor --no-md / drop --json sidecar when not asked for.
        for src in args.inputs:
            _reconcile_outputs(
                src,
                diarize=False,
                srt=False,
                json_out=args.json,
                no_md=args.no_md,
            )

    from speech2md._gpu import silence_teardown
    silence_teardown()
    return rc


def _run_pipeline(args) -> int:
    """Orchestration path (when --srt or --diarize): run transcribe,
    align, and optionally diarize as separate subprocesses so each
    model gets a clean GPU, then reconcile outputs to match flags.

    Batch semantics: every stage handles all inputs in one subprocess
    call, so each heavy model is loaded exactly once per stage."""
    # Phase 1: prose ASR with larger chunks.
    prose_cmd = _transcribe_stage_cmd(
        args,
        keep_json=False,
        no_md=False,
        target=args.target,
        max_chunk=args.max_chunk,
    )
    if subprocess.call(prose_cmd) != 0:
        return 1

    # Phase 2: alignment-prep ASR with shorter chunks. This pass only keeps
    # the JSON sidecar for the forced aligner, then the parent restores the
    # prose markdown from the long-chunk pass.
    saved_md = _snapshot_markdown(args.inputs)
    align_prep_cmd = _transcribe_stage_cmd(
        args,
        keep_json=True,
        no_md=True,
        target=args.align_target,
        max_chunk=args.align_max_chunk,
    )
    try:
        if subprocess.call(align_prep_cmd) != 0:
            return 1
    finally:
        _restore_markdown(saved_md)

    # Phase 3: align. Only feed in files whose chunks JSON was actually
    # written — a prior failure in phase 1 (or --skip-existing) can
    # leave a subset.
    chunks_jsons = [
        src.with_name(f"{src.stem}.json") for src in args.inputs
        if src.with_name(f"{src.stem}.json").exists()
    ]
    if not chunks_jsons:
        return 1
    align_cmd = [
        sys.executable, "-m", "speech2md.align",
        *(str(x) for x in chunks_jsons),
    ]
    if subprocess.call(align_cmd) != 0:
        return 1

    # Phase 4: diarize (only when --diarize).
    if args.diarize:
        words_jsons = [
            src.with_name(f"{src.stem}.words.json") for src in args.inputs
            if src.with_name(f"{src.stem}.words.json").exists()
        ]
        if not words_jsons:
            return 1
        diarize_cmd = [
            sys.executable, "-m", "speech2md.diarize",
            *(str(x) for x in words_jsons),
        ]
        if args.num_speakers is not None:
            diarize_cmd += ["--num-speakers", str(args.num_speakers)]
        if args.min_speakers is not None:
            diarize_cmd += ["--min-speakers", str(args.min_speakers)]
        if args.max_speakers is not None:
            diarize_cmd += ["--max-speakers", str(args.max_speakers)]
        if args.max_gap is not None:
            diarize_cmd += ["--max-gap", str(args.max_gap)]
        if args.hf_token:
            diarize_cmd += ["--token", args.hf_token]
        if subprocess.call(diarize_cmd) != 0:
            return 1

    # Phase 5: per-file rename / cleanup to match requested outputs.
    for src in args.inputs:
        _reconcile_outputs(
            src,
            diarize=args.diarize,
            srt=args.srt,
            json_out=args.json,
            no_md=args.no_md,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
