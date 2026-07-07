# Fun-ASR-Nano 实时 WebSocket 服务 — 快速上手

> 完整文档请参见：[FunASR vLLM 推理引擎指南](https://github.com/modelscope/FunASR/blob/main/docs/vllm_guide.md)

> **注意**: `serve_realtime_ws.py` 等服务脚本位于 [FunASR 主仓库](https://github.com/modelscope/FunASR/tree/main/examples/industrial_data_pretraining/fun_asr_nano)，需要先 clone FunASR。

## 30 秒启动

```bash
# Clone FunASR main repo (contains the server scripts)
git clone https://github.com/modelscope/FunASR.git
cd FunASR/examples/industrial_data_pretraining/fun_asr_nano

# 安装依赖
pip install -r requirements.txt
pip install vllm>=0.12.0

# 启动服务
CUDA_VISIBLE_DEVICES=0 python serve_realtime_ws.py \
    --port 10095 --language 中文 \
    --ws-ping-interval 30 --ws-ping-timeout 120
```

长时间会议、访谈或经过 Nginx/负载均衡访问时，可按网络空闲回收时间调整 `--ws-ping-interval` 与 `--ws-ping-timeout`。如果外部网关已经负责心跳和重连，可将两者设为 `0` 关闭 WebSocket 协议层 ping。

## 客户端

```bash
# 浏览器
open client_mic.html

# Python 麦克风
python client_python.py --server ws://localhost:10095 --mic

# Python 文件
python client_python.py --server ws://localhost:10095 --file audio.wav

# 自动化测试
python client_test.py --server ws://localhost:10095 --file audio.wav
```

## 远程访问

```bash
ssh -L 10095:localhost:10095 <server>
# 然后本地打开 client_mic.html
```

## 功能

- **vLLM 推理引擎**：RTF < 0.08，支持 tensor parallel 多卡加速
- **流式 VAD 分句**：动态静音阈值，自然断句
- **说话人分离 (Beta)**：流式 ID 分配 + 最终重聚类
- **热词定制化**：加载人名、地名等实体词文件
- **语种指定**：31 种语言 + 中文方言
- **幻觉检测**：自动检测重复模式并截断

## 文件列表

| 文件 | 说明 |
|------|------|
| `serve_realtime_ws.py` | WebSocket 服务端 |
| `client_mic.html` | 浏览器客户端 |
| `client_python.py` | Python CLI 客户端 |
| `client_test.py` | 自动化测试脚本 |
| `热词列表` | 热词文件示例 |
| `demo_vllm.py` | 离线 vLLM 推理 demo |

## 详细文档

- [FunASR vLLM 推理引擎指南](https://github.com/modelscope/FunASR/blob/main/docs/vllm_guide.md) — 完整文档（离线/流式/WebSocket/API/FAQ）
