# FunASR Real-time VAD + ASR Demo

这是一个基于 FunASR 和 Silero VAD 实现的实时语音识别 Demo。包含命令行（CLI）版本和 Web 版本。

## 特性 (Features)

*   **实时语音活动检测 (VAD)**: 集成 Silero VAD，精准检测语音片段。
*   **流式识别**: 语音检测期间实时更新部分结果 (Partial)，句子结束（静音检测）后提交最终结果 (Final)。
*   **Apple Silicon 优化**: 自动检测并使用 MPS (Metal Performance Shaders) 加速推理。
*   **Web 可视化**: 提供无闪烁的 Web 界面展示实时转写结果，包含 VAD 概率曲线与 ASR 处理耗时标记。
*   **双模式输入**: 支持麦克风实时输入和音频文件模拟输入。

## 核心逻辑与状态机 (Core Logic & State Machine)

本 Demo 采用基于 VAD 的状态机来控制 ASR 识别流程，以平衡实时性与计算开销。

### 1. 状态：静音监听 (Listening / Silence)
*   **行为**: 持续读取音频流 (Chunk size: 512 samples, ~32ms)。
*   **Pre-roll 缓冲**: 维护一个长度为 3 的循环缓冲区 (`pre_roll`)，存储最近 ~100ms 的静音片段。
*   **状态迁移**:
    *   当 VAD 概率 `> 0.5` 时 -> 进入 **语音活动 (Speech Active)** 状态。
    *   **Action**: 将 `pre_roll` 中的数据 + 当前 Chunk 一起放入 `speech_buffer`。
    *   *目的*: 确保语音的起始音节（如爆破音）不被 VAD 阈值截断。

### 2. 状态：语音活动 (Speech Active)
在此状态下，系统持续收集音频数据并根据策略触发识别。

*   **行为**: 将后续的音频 Chunk 持续追加到 `speech_buffer`。
    *   **增量更新策略 (Batch Processing)**:
    *   为了减少 GPU/CPU 负载，不会对每个 Chunk 都进行识别。
    *   **逻辑**: 维护 `speech_chunks_since_update` 计数器。
    *   当积累满 **30 个 Chunks** (约 0.96秒) 时 -> 触发一次 **Partial ASR**。
    *   *效果*: 用户每隔约 1 秒看到一次字幕更新，而不是逐字跳动。

### 3. 状态：句子结束判定 (Sentence Finalization)
*   **静音检测**:
    *   在 **语音活动** 状态中，如果当前 Chunk 的 VAD 概率 `< 0.5`，则认为可能是静音。
    *   累加 `silence_counter`。
    *   一旦检测到语音 (VAD `> 0.5`)，立即重置 `silence_counter` 为 0（容忍句子中间的短暂停顿）。
*   **状态迁移**:
    *   当 `silence_counter` 持续时长超过 **500ms** -> 判定为句子结束。    *   **Action**:
        1. 将 `speech_buffer` 中的完整音频提交进行 **Final ASR**。
        2. 清空 `speech_buffer`。
        3. 重置所有计数器。
        4. 返回 **静音监听** 状态。

## 依赖安装 (Requirements)

请确保项目根目录下的依赖已安装，并额外安装以下库：

```bash
pip install pyaudio fastapi uvicorn websockets
```

*   **PyAudio 注意事项**:
    *   **macOS**: 如果安装失败，需先安装 portaudio: `brew install portaudio`
    *   **Linux**: `sudo apt-get install python3-pyaudio`

## 使用说明 (Usage)

**注意**: 请在项目根目录下运行以下命令，以确保正确加载 `model.py`。

### 1. 命令行演示 (CLI Demo)

直接在终端输出识别结果。

**麦克风输入:**
```bash
python vad_demo/demo_vad.py
```

**文件模拟输入:**
```bash
python vad_demo/demo_vad.py --file /path/to/your/audio.wav
```

### 2. 网页演示 (Web Demo)

启动 Web 服务，在浏览器中查看实时字幕及 VAD 可视化。

**启动服务 (麦克风):**
```bash
python vad_demo/web_demo.py
```

**启动服务 (文件模拟):**
```bash
python vad_demo/web_demo.py --file /path/to/your/audio.wav
```

启动后，在浏览器访问: [http://localhost:8000](http://localhost:8000)

### 3. VAD 模型测试

检测 Silero VAD 模型是否能正确加载及使用 MPS 加速。

```bash
python vad_demo/test_silero.py
```

## 目录结构

*   `demo_vad.py`: 核心逻辑实现，CLI 版本入口。
*   `web_demo.py`: 基于 FastAPI 的 Web 服务入口，包含可视化前端。
*   `test_silero.py`: VAD 环境测试脚本。