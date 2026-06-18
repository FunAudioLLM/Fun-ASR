# Fun-ASR-Nano on llama.cpp / GGUF

Run **Fun-ASR-Nano** entirely on the [llama.cpp](https://github.com/ggml-org/llama.cpp) /
ggml stack — CPU, edge, single binary, no Python at runtime. This is to Fun-ASR
what whisper.cpp is to Whisper.

Fun-ASR-Nano = SenseVoice SAN-M encoder + adaptor + Qwen3-0.6B LLM decoder. The
whole pipeline runs in C++:

```
WAV (16k mono) → kaldi fbank → SAN-M encoder + adaptor (ggml) → low-frame-rate
truncation → [prefix tokens | audio embeds | suffix tokens] → Qwen3 (llama.cpp) → text
```

The audio embeddings are injected into the LLM via `llama_decode`'s embedding
input path — the same mechanism llava/mtmd use for vision embeddings.

## Contents
- `funasr-cli/`     — integrated single binary: WAV → transcription
- `funasr-encoder/` — encoder+adaptor only (ggml), for validation/debugging
- `funasr-embd/`    — LLM decode from precomputed embeds, for validation
- `export_encoder_gguf.py` — export the audio encoder + adaptor to GGUF

## Build
```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cp -r /path/to/funasr-cli examples/
echo 'add_subdirectory(funasr-cli)' >> examples/CMakeLists.txt
cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build -j --target llama-funasr-cli
```

## Convert weights to GGUF (one-time)
```bash
# LLM half: extract Qwen3-0.6B (HF format under the model dir) and convert
python llama.cpp/convert_hf_to_gguf.py <model>/Qwen3-0.6B-vllm \
    --outfile qwen3-0.6b-f32.gguf --outtype f32
build/bin/llama-quantize qwen3-0.6b-f32.gguf qwen3-0.6b-q8_0.gguf Q8_0   # optional

# audio half: SenseVoice encoder + adaptor
python export_encoder_gguf.py --model_pt <model>/model.pt \
    --out funasr-encoder.gguf            # f32 (935 MB)
python export_encoder_gguf.py --model_pt <model>/model.pt \
    --out funasr-encoder-f16.gguf --wtype f16   # 469 MB
```

## Transcribe
```bash
build/bin/llama-funasr-cli \
    --enc funasr-encoder.gguf \
    -m   qwen3-0.6b-q8_0.gguf \
    -a   audio.wav \
    --chunk 15           # split long audio into 15s windows (recommended)
```

## Notes & validation
- The 70-layer SAN-M encoder + adaptor were validated against PyTorch:
  encoder cosine **1.0**, max_abs_diff **5e-3** (f32). The kaldi fbank front end
  matches torchaudio (cosine 1.0). Under identical conditions (f32 LLM, same
  segmentation) the C++ pipeline's aggregate CER matches PyTorch to within 0.02%.
- FSMN depthwise memory is implemented as an exact f32 shift-accumulate.
- LayerNorm eps 1e-5; sinusoidal position encoding depth = input feature dim (560),
  positions start at 1, input pre-scaled by sqrt(512).
- Low-frame-rate: only the first `fake_token_len` adaptor frames are used as audio
  tokens (the CLI handles this); feeding all frames is out-of-distribution.
- WAV input currently assumes 16 kHz mono PCM16.

Requires a Fun-ASR-Nano checkpoint (e.g. `FunAudioLLM/Fun-ASR-Nano-2512`).
