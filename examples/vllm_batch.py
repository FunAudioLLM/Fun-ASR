#!/usr/bin/env python3
"""Offline batch inference with funasr.auto.auto_model_vllm.AutoModelVLLM."""

import argparse
from pathlib import Path


def resolve_audio_paths(audio_files: list[str], model_path: str) -> list[str]:
    if audio_files:
        paths = [Path(audio).expanduser() for audio in audio_files]
    else:
        paths = [Path(model_path) / "example" / "zh.mp3"]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Audio file(s) not found: {missing}")
    return [str(path) for path in paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio",
        nargs="*",
        help="Audio file(s). Defaults to model example/zh.mp3",
    )
    parser.add_argument("--model", default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--language", default="中文")
    parser.add_argument(
        "--hotwords",
        nargs="*",
        default=[],
        help="Optional hotwords, separated by spaces",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    from funasr.auto.auto_model_vllm import AutoModelVLLM

    args = parse_args()
    model = AutoModelVLLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    model_path = getattr(model, "model_dir", args.model)
    audio_files = resolve_audio_paths(args.audio, model_path)

    results = model.generate(
        audio_files,
        language=args.language,
        hotwords=args.hotwords,
    )
    for result in results:
        key = result.get("key", "audio")
        print(f"[{key}] {result['text']}")


if __name__ == "__main__":
    main()
