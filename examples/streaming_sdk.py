#!/usr/bin/env python3
"""Stream an audio file with FunASRNanoStreamingVLLM."""

import argparse
from pathlib import Path


def resolve_audio_path(audio: str | None, model_path: str) -> str:
    if audio:
        path = Path(audio).expanduser()
    else:
        path = Path(model_path) / "example" / "zh.mp3"
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", help="Audio file. Defaults to model example/zh.mp3")
    parser.add_argument("--model", default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--hub", default="hf", choices=["hf", "ms"])
    parser.add_argument("--language", default="中文")
    parser.add_argument("--chunk-ms", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    from funasr.models.fun_asr_nano.inference_vllm_streaming import (
        FunASRNanoStreamingVLLM,
    )

    args = parse_args()
    engine = FunASRNanoStreamingVLLM.from_pretrained(
        model=args.model,
        hub=args.hub,
        chunk_ms=args.chunk_ms,
    )
    model_path = getattr(engine, "model_dir", args.model)
    wav_path = resolve_audio_path(args.audio, model_path)

    for result in engine.streaming_generate(wav_path, language=args.language):
        duration_ms = result.get("audio_duration_ms", 0.0)
        fixed_text = result.get("fixed_text", "")
        suffix = " final" if result.get("is_final") else ""
        print(f"[{duration_ms:.0f}ms]{suffix} {fixed_text}")


if __name__ == "__main__":
    main()
