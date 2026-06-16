# Fun-ASR

「[简体中文](README_zh.md)」|「[English](README.md)」|「日本語」

Fun-ASRは通義実験室が開発したエンドツーエンド音声認識大規模モデルです。数千万時間の実音声データで学習され、強力なコンテキスト理解能力と業界適応性を備えています。低遅延リアルタイム文字起こしをサポートし、31言語に対応しています。

<div align="center">
<img src="images/funasr-v2.png">
</div>

<div align="center">
<h4>
<a href="https://funaudiollm.github.io/funasr"> ホームページ </a>
｜<a href="#主要機能"> 主要機能 </a>
｜<a href="#性能評価"> 性能評価 </a>
｜<a href="#環境構築"> 環境構築 </a>
｜<a href="#使い方"> 使い方 </a>

</h4>

モデルリポジトリ：[ModelScope](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512)、[HuggingFace](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)

オンラインデモ：
[ModelScope Space](https://modelscope.cn/studios/FunAudioLLM/Fun-ASR-Nano)、[HuggingFace Space](https://huggingface.co/spaces/FunAudioLLM/Fun-ASR-Nano)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FunAudioLLM/Fun-ASR/blob/main/examples/colab/fun_asr_nano_quickstart.ipynb)

</div>

| モデル | 対応タスク | 学習データ | パラメータ |
| :---: | :---: | :---: | :---: |
| Fun-ASR-Nano <br> ([⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)) | 中国語・英語・日本語の音声認識。中国語は7方言・26地域アクセント対応。英語・日本語も複数地域アクセントに対応。歌詞認識・ラップ音声認識も搭載。 | 数千万時間 | 8億 |
| Fun-ASR-MLT-Nano <br> ([⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512)) | 韓国語、ベトナム語、インドネシア語、タイ語、マレー語、フィリピン語、アラビア語、ヒンディー語など31言語の音声認識。 | 数十万時間 | 8億 |

<a name="主要機能"></a>

# 主要機能 🎯

- **遠距離・高ノイズ環境対応**：会議室、車内、工場など高ノイズ環境に最適化、認識精度 **93%** 達成
- **中国語方言・地域アクセント**：7大方言 + 26地域アクセントに対応
- **31言語多言語対応**：東南アジア言語に重点最適化、言語自動切替・混合認識対応
- **音楽背景下の歌詞認識**：音楽干渉下での音声認識性能を強化
- **ホットワード機能**：ドメイン固有用語の認識精度を向上
- **話者分離**：誰がいつ話したかを自動識別
- **vLLM推論エンジン**：バッチ推論で最大340倍リアルタイム速度

<a name="環境構築"></a>

# 環境構築 🐍

```shell
git clone https://github.com/FunAudioLLM/Fun-ASR.git
cd Fun-ASR
pip install -r requirements.txt
```

<a name="使い方"></a>

# 使い方 🛠️

## 基本的な推論

```python
from funasr import AutoModel

model = AutoModel(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    trust_remote_code=True,
    device="cuda:0",
    hub="hf"
)

result = model.generate(
    input=["audio.wav"],
    batch_size=1,
    language="日文",
)
print(result[0]["text"])
```

## 話者分離付き

```python
model = AutoModel(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    trust_remote_code=True,
    device="cuda:0",
    hub="hf",
    vad_model="fsmn-vad",
    spk_model="cam++",
    punc_model="ct-punc"
)

result = model.generate(input=["meeting.wav"], batch_size=1)
for item in result:
    if 'sentence_info' in item:
        for sent in item['sentence_info']:
            print(f"[話者{sent['spk']}] {sent['text']}")
```

## vLLM 高速推論

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

model = AutoModelVLLM(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    tensor_parallel_size=2,
)

results = model.generate(["audio1.wav", "audio2.wav"], language="日文")
```

詳細は [vLLM推論ガイド](docs/vllm_guide.md) をご参照ください。

<a name="性能評価"></a>

# 性能評価 📊

| モデル | GPUスピード | CPUスピード | vs Whisper-large-v3 |
|--------|-----------|-----------|-------------------|
| Fun-ASR-Nano (vLLM) | **340x** リアルタイム | — | 🚀 **26倍高速** |
| SenseVoice-Small | **170x** リアルタイム | **17x** リアルタイム | 🚀 **13倍高速** |
| Whisper-large-v3 | 13x リアルタイム | ❌ | 基準 |

## エコシステム

Fun-ASR-Nanoは **FunAudioLLM** ファミリーの一員です：

| プロジェクト | 説明 | Stars |
|-------------|------|-------|
| [FunASR](https://github.com/modelscope/FunASR) | 産業用音声認識ツールキット — VAD、ASR、句読点、話者分離 | [![](https://img.shields.io/github/stars/modelscope/FunASR?style=social)](https://github.com/modelscope/FunASR) |
| [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) | 超高速ASR + 感情認識 + 音声イベント検出 | [![](https://img.shields.io/github/stars/FunAudioLLM/SenseVoice?style=social)](https://github.com/FunAudioLLM/SenseVoice) |
| [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | 自然音声生成 — 多言語、ゼロショットクローニング | [![](https://img.shields.io/github/stars/FunAudioLLM/CosyVoice?style=social)](https://github.com/FunAudioLLM/CosyVoice) |
| [FunClip](https://github.com/modelscope/FunClip) | AI音声認識による動画クリッピング | [![](https://img.shields.io/github/stars/modelscope/FunClip?style=social)](https://github.com/modelscope/FunClip) |

## ライセンス

[Apache 2.0](LICENSE)
