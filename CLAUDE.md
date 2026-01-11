基于 demo1.py，结合 silero-vad，实现一个 python+vad demo。只要 VAD 检测到语音，不需要等结束，即开始实时更新 ASR 结果：按 chunk 进 queue 再进行 ASR，当有更多 chunk 时和 queue 里 chunk 合并再 ASR 并覆盖之前 ASR 结果。如果形成了完整的句子，则新开 queue，并提交当前 ASR 结果。
尽量用 apple silicon MPS 运行。
