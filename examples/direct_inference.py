#!/usr/bin/env python3
"""Run Fun-ASR-Nano by importing this repository's model.py directly."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    parser.add_argument(
        "--device",
        default="auto",
        help="Device such as cuda:0, cpu, mps, or auto",
    )
    return parser.parse_args()


def main() -> None:
    from model import FunASRNano

    args = parse_args()
    device = pick_device(args.device)
    model, kwargs = FunASRNano.from_pretrained(
        model=args.model,
        device=device,
        hub=args.hub,
    )
    model.eval()

    wav_path = resolve_audio_path(args.audio, kwargs["model_path"])
    res = model.inference(data_in=[wav_path], **kwargs)
    print(res[0][0]["text"])


if __name__ == "__main__":
    main()
