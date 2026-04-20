"""Word-level timestamps for a transcript from `speech2md`.

Runs in its own process (vLLM + forced aligner together OOM on 24 GB).
Uses `Qwen3ForcedAligner` directly — a standalone 0.6B model, ~1–2 GB VRAM.
Input: the JSON sidecar from `speech2md`. Output: a words JSON
and an SRT file.

Usage:
    align-transcription audio.json   # writes audio.words.json + audio.srt
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def cut_chunk(src: Path, start: float, end: float, out: Path) -> None:
    subprocess.check_call(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out)]
    )


def fmt_srt_ts(seconds: float) -> str:
    ms = int(seconds * 1000)
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_ts(seconds: float) -> str:
    td = dt.timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    m, s = divmod(rem, 60)
    h += td.days * 24
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json_in", type=Path, help="JSON sidecar from speech2md")
    p.add_argument("--audio", type=Path, help="source audio path (default: from json 'source')")
    p.add_argument("--aligner", default="Qwen/Qwen3-ForcedAligner-0.6B")
    p.add_argument("-o", "--out-json", type=Path, help="word-level json (default: <input>.words.json)")
    p.add_argument("--srt", type=Path, help="SRT output path (default: <input>.srt)")
    p.add_argument("--keep-chunks", action="store_true")
    args = p.parse_args()

    data = json.loads(args.json_in.read_text(encoding="utf-8"))
    audio_path = args.audio or Path(data["source"])
    if not audio_path.exists():
        print(f"audio not found: {audio_path}", file=sys.stderr)
        return 2

    from speech2md._gpu import require_forced_aligner, require_torch, silenced_stderr
    torch = require_torch()
    Qwen3ForcedAligner = require_forced_aligner()

    print(f"loading {args.aligner} ...")
    with silenced_stderr():
        aligner = Qwen3ForcedAligner.from_pretrained(
            args.aligner,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )

    workdir = Path(tempfile.mkdtemp(prefix="speech2md-align-"))
    try:
        words_out: list[dict] = []  # flat list across all chunks
        for i, c in enumerate(data["chunks"], 1):
            text = (c.get("text") or "").strip()
            if not text:
                print(f"[{i}/{len(data['chunks'])}] skip (empty)")
                continue
            wav = workdir / f"chunk_{i:02d}.wav"
            cut_chunk(audio_path, c["start"], c["end"], wav)
            lang = c.get("language") or data.get("language") or "English"
            try:
                results = aligner.align(audio=str(wav), text=text, language=lang)
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(data['chunks'])}] align failed: {e}", file=sys.stderr)
                continue
            r = results[0] if isinstance(results, list) else results
            segs = getattr(r, "items", None) or getattr(r, "time_stamps", None) or []
            for seg in segs:
                st = getattr(seg, "start_time", None)
                et = getattr(seg, "end_time", None)
                tx = getattr(seg, "text", "") or ""
                if st is None or et is None or not tx.strip():
                    continue
                words_out.append({
                    "start": round(c["start"] + float(st), 3),
                    "end": round(c["start"] + float(et), 3),
                    "text": tx,
                })
            print(f"[{i}/{len(data['chunks'])}] {fmt_ts(c['start'])}  {len(segs)} segments")

        out_json = args.out_json or args.json_in.with_name(args.json_in.stem + ".words.json")
        out_srt = args.srt or args.json_in.with_suffix(".srt")

        out_json.write_text(json.dumps({
            **{k: v for k, v in data.items() if k != "chunks"},
            "words": words_out,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        with out_srt.open("w", encoding="utf-8") as srt:
            for i, w in enumerate(words_out, 1):
                srt.write(f"{i}\n{fmt_srt_ts(w['start'])} --> {fmt_srt_ts(w['end'])}\n{w['text']}\n\n")

        print(f"wrote {out_json.name} ({len(words_out)} words) and {out_srt.name}")
        rc = 0
    finally:
        if not args.keep_chunks:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"chunks kept at {workdir}")

    from speech2md._gpu import silence_teardown
    silence_teardown()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
