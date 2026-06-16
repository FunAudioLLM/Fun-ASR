# Fun-ASR

「[简体中文](README_zh.md)」|「[English](README.md)」|「[日本語](README_ja.md)」|「한국어」

Fun-ASR는 통의(Tongyi) 실험실에서 출시한 엔드투엔드 음성 인식 대규모 모델입니다. 수천만 시간의 실제 음성 데이터로 학습되었으며, 강력한 문맥 이해 능력과 산업 적응성을 갖추고 있습니다. 저지연 실시간 전사를 지원하며 31개 언어를 포함합니다.

<div align="center">
<img src="images/funasr-v2.png">
</div>

<div align="center">
<h4>
<a href="https://funaudiollm.github.io/funasr"> 홈페이지 </a>
｜<a href="#주요-기능"> 주요 기능 </a>
｜<a href="#성능-평가"> 성능 평가 </a>
｜<a href="#환경-설정"> 환경 설정 </a>
｜<a href="#사용법"> 사용법 </a>

</h4>

모델 저장소: [ModelScope](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512), [HuggingFace](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)

온라인 체험:
[ModelScope Space](https://modelscope.cn/studios/FunAudioLLM/Fun-ASR-Nano), [HuggingFace Space](https://huggingface.co/spaces/FunAudioLLM/Fun-ASR-Nano)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FunAudioLLM/Fun-ASR/blob/main/examples/colab/fun_asr_nano_quickstart.ipynb)

</div>

| 모델 | 지원 작업 | 학습 데이터 | 파라미터 |
| :---: | :---: | :---: | :---: |
| Fun-ASR-Nano <br> ([⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)) | 중국어·영어·일본어 음성 인식. 중국어 7개 방언 + 26개 지역 억양 지원. 영어·일본어도 다양한 억양 대응. 가사 인식·랩 음성 인식 탑재. | 수천만 시간 | 8억 |
| Fun-ASR-MLT-Nano <br> ([⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512)) | 한국어, 베트남어, 인도네시아어, 태국어, 말레이어, 필리핀어, 아랍어, 힌디어 등 31개 언어 음성 인식. | 수십만 시간 | 8억 |

<a name="주요-기능"></a>

# 주요 기능 🎯

- **원거리·고소음 환경 대응**: 회의실, 차량, 공장 등 고소음 환경에 최적화, 인식 정확도 **93%** 달성
- **31개 언어 다국어 지원**: 동남아시아 언어에 중점 최적화, 자동 언어 전환·혼합 인식 지원
- **한국어 지원**: Fun-ASR-MLT-Nano를 통한 한국어 음성 인식
- **핫워드 기능**: 도메인 특정 용어의 인식 정확도 향상
- **화자 분리**: 누가 언제 말했는지 자동 식별
- **vLLM 추론 엔진**: 배치 추론으로 최대 340배 실시간 속도

<a name="환경-설정"></a>

# 환경 설정 🐍

```shell
git clone https://github.com/FunAudioLLM/Fun-ASR.git
cd Fun-ASR
pip install -r requirements.txt
```

<a name="사용법"></a>

# 사용법 🛠️

## 기본 추론

```python
from funasr import AutoModel

model = AutoModel(
    model="FunAudioLLM/Fun-ASR-MLT-Nano-2512",  # 한국어는 MLT 모델 사용
    trust_remote_code=True,
    device="cuda:0",
    hub="hf"
)

result = model.generate(
    input=["audio.wav"],
    batch_size=1,
    language="韩文",
)
print(result[0]["text"])
```

## 화자 분리 포함

```python
model = AutoModel(
    model="FunAudioLLM/Fun-ASR-MLT-Nano-2512",
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
            print(f"[화자{sent['spk']}] {sent['text']}")
```

<a name="성능-평가"></a>

# 성능 평가 📊

| 모델 | GPU 속도 | CPU 속도 | vs Whisper-large-v3 |
|------|---------|---------|-------------------|
| Fun-ASR-Nano (vLLM) | **340x** 실시간 | — | 🚀 **26배 빠름** |
| SenseVoice-Small | **170x** 실시간 | **17x** 실시간 | 🚀 **13배 빠름** |
| Whisper-large-v3 | 13x 실시간 | ❌ | 기준 |

## 에코시스템

Fun-ASR-Nano는 **FunAudioLLM** 패밀리의 일원입니다:

| 프로젝트 | 설명 | Stars |
|----------|------|-------|
| [FunASR](https://github.com/modelscope/FunASR) | 산업용 음성 인식 툴킷 — VAD, ASR, 구두점, 화자 분리 | [![](https://img.shields.io/github/stars/modelscope/FunASR?style=social)](https://github.com/modelscope/FunASR) |
| [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) | 초고속 ASR + 감정 인식 + 오디오 이벤트 감지 | [![](https://img.shields.io/github/stars/FunAudioLLM/SenseVoice?style=social)](https://github.com/FunAudioLLM/SenseVoice) |
| [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | 자연 음성 생성 — 다국어, 제로샷 클로닝 | [![](https://img.shields.io/github/stars/FunAudioLLM/CosyVoice?style=social)](https://github.com/FunAudioLLM/CosyVoice) |
| [FunClip](https://github.com/modelscope/FunClip) | AI 음성 인식 기반 비디오 클리핑 | [![](https://img.shields.io/github/stars/modelscope/FunClip?style=social)](https://github.com/modelscope/FunClip) |

## 라이선스

[Apache 2.0](LICENSE)
