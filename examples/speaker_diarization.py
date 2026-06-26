#!/usr/bin/env python3
"""Run Fun-ASR-Nano with VAD, punctuation, and speaker diarization."""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--language", default="中文")
    return parser.parse_args()


def main() -> None:
    from funasr import AutoModel

    args = parse_args()
    device = pick_device(args.device)
    model = AutoModel(
        model=args.model,
        trust_remote_code=True,
        remote_code=str(ROOT / "model.py"),
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        spk_model="cam++",
        punc_model="ct-punc",
        device=device,
        hub=args.hub,
    )

    wav_path = resolve_audio_path(args.audio, model.model_path)
    res = model.generate(input=[wav_path], cache={}, batch_size=1, language=args.language)
    for sent in res[0]["sentence_info"]:
        print(
            f"Speaker {sent['spk']}: "
            f"[{sent['start']}ms - {sent['end']}ms] {sent['sentence']}"
        )


if __name__ == "__main__":
    main()
