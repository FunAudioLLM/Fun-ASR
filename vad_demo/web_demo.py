import os
import sys
import queue
import threading
import asyncio
import json
import time
import argparse
import logging
import numpy as np
import torch
import torchaudio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from typing import List

# Suppress warnings and logs
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('funasr').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

# Try importing pyaudio
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

from funasr import AutoModel

# Configuration
CHUNKSZ = 512  # 32ms at 16kHz
RATE = 16000
VAD_THRESHOLD = 0.5
SILENCE_DURATION_MS = 500
MIN_SPEECH_MS = 200

# HTML Template
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>FunASR Real-time Demo</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background-color: #f0f2f5; color: #333; }
            #container { background-color: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
            h1 { color: #1a1a1a; text-align: center; margin-bottom: 20px; font-weight: 600; }
            
            #status { text-align: center; font-weight: 500; margin-bottom: 20px; padding: 8px; border-radius: 6px; }
            .status-connected { background-color: #e6fffa; color: #047857; border: 1px solid #047857; }
            .status-disconnected { background-color: #fff5f5; color: #c53030; border: 1px solid #c53030; }
            
            #viz-container { margin-bottom: 25px; position: relative; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; background: #fff; }
            canvas { display: block; width: 100%; height: 200px; }
            .legend { display: flex; justify-content: center; gap: 20px; margin-top: 5px; font-size: 0.85em; color: #666; }
            .legend-item { display: flex; align-items: center; gap: 5px; }
            .dot { width: 10px; height: 10px; border-radius: 50%; }
            
            #results-area { display: flex; flex-direction: column; gap: 20px; }
            
            .box { border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; }
            .box-header { background: #f9fafb; padding: 10px 15px; border-bottom: 1px solid #e5e7eb; font-weight: 600; font-size: 0.9em; color: #4b5563; text-transform: uppercase; letter-spacing: 0.05em; }
            
            #partial-container { padding: 20px; min-height: 60px; display: flex; align-items: center; background: #ffffff; font-size: 1.1em; color: #1f2937; }
            .placeholder { color: #9ca3af; font-style: italic; }
            
            #final-results { height: 350px; overflow-y: auto; padding: 0; background: #ffffff; }
            .sentence { padding: 12px 15px; border-bottom: 1px solid #f3f4f6; display: flex; gap: 10px; animation: fadeIn 0.3s ease; }
            .sentence:last-child { border-bottom: none; }
            .timestamp { color: #9ca3af; font-size: 0.85em; font-variant-numeric: tabular-nums; flex-shrink: 0; padding-top: 2px; }
            .text { line-height: 1.5; }
            
            @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
            
            /* Scrollbar styling */
            #final-results::-webkit-scrollbar { width: 8px; }
            #final-results::-webkit-scrollbar-track { background: #f1f1f1; }
            #final-results::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 4px; }
            #final-results::-webkit-scrollbar-thumb:hover { background: #a8a8a8; }
        </style>
    </head>
    <body>
        <div id="container">
            <h1>FunASR Live Demo</h1>
            <div id="status" class="status-disconnected">Disconnected</div>
            
            <div id="viz-container">
                <canvas id="vadCanvas" width="800" height="200"></canvas>
            </div>
            <div class="legend">
                <div class="legend-item"><span class="dot" style="background:#2563eb"></span> VAD Probability</div>
                <div class="legend-item"><span class="dot" style="background:rgba(255, 165, 0, 0.3)"></span> ASR Processing</div>
            </div>
            <br>

            <div id="results-area">
                <div class="box">
                    <div class="box-header">Current Input</div>
                    <div id="partial-container">
                        <span id="partial" class="placeholder">Listening...</span>
                    </div>
                </div>
                
                <div class="box">
                    <div class="box-header" style="display: flex; justify-content: space-between; align-items: center;">
                        History
                        <button onclick="copyHistory()" style="padding: 4px 8px; font-size: 0.8em; cursor: pointer;">Copy JSON</button>
                    </div>
                    <div id="final-results"></div>
                </div>
            </div>
        </div>

        <script>
            const ws = new WebSocket("ws://" + location.host + "/ws");
            const canvas = document.getElementById("vadCanvas");
            const ctx = canvas.getContext("2d");
            
            // Data buffers
            const TIME_WINDOW = 10000; // 10 seconds history
            let vadData = []; // {t: timestamp, v: value}
            let asrEvents = []; // {start: timestamp, end: timestamp|null}
            let currentAsrStart = null;
            
            // UI Elements
            const partialSpan = document.getElementById("partial");
            const finalResultsDiv = document.getElementById("final-results");
            const statusDiv = document.getElementById("status");
            
            // History Data
            let historyData = [];

            function copyHistory() {
                const jsonStr = JSON.stringify(historyData, null, 2);
                navigator.clipboard.writeText(jsonStr).then(() => {
                    alert("History copied to clipboard!");
                }).catch(err => {
                    console.error("Failed to copy text: ", err);
                });
            }

            ws.onopen = () => {
                statusDiv.innerText = "Connected to Server";
                statusDiv.className = "status-connected";
                requestAnimationFrame(drawLoop);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                const now = Date.now();

                if (data.type === "vad") {
                    vadData.push({t: now, v: data.prob});
                    // Prune old data
                    if (vadData.length > 0 && now - vadData[0].t > TIME_WINDOW + 1000) {
                        vadData.shift();
                    }
                } 
                else if (data.type === "asr_start") {
                    currentAsrStart = now;
                }
                else if (data.type === "partial" || data.type === "final") {
                    // Text Update
                    if (data.type === "partial") {
                        partialSpan.className = "";
                        partialSpan.innerText = data.text;
                    } else {
                        partialSpan.innerText = "";
                        addFinalResult(data.text);
                    }
                    
                    // Close ASR event interval
                    if (currentAsrStart !== null) {
                        asrEvents.push({start: currentAsrStart, end: now});
                        currentAsrStart = null;
                    }
                }
            };
            
            function addFinalResult(text) {
                const now = new Date();
                const timeStr = now.toLocaleTimeString();
                
                historyData.push({timestamp: now.toISOString(), text: text});
                
                const div = document.createElement("div");
                div.className = "sentence";
                div.innerHTML = `
                    <span class="timestamp">${timeStr}</span>
                    <span class="text">${text}</span>
                `;
                finalResultsDiv.appendChild(div);
                finalResultsDiv.scrollTop = finalResultsDiv.scrollHeight;
            }

            ws.onclose = () => {
                statusDiv.innerText = "Disconnected";
                statusDiv.className = "status-disconnected";
            };

            // Visualization Loop
            function drawLoop() {
                const now = Date.now();
                const width = canvas.width;
                const height = canvas.height;
                
                ctx.clearRect(0, 0, width, height);
                
                // 1. Draw Grid / Background
                ctx.strokeStyle = "#f0f0f0";
                ctx.lineWidth = 1;
                ctx.beginPath();
                for (let i = 0; i < 5; i++) {
                    let y = height * (i/4);
                    ctx.moveTo(0, y);
                    ctx.lineTo(width, y);
                }
                ctx.stroke();
                
                // Helper to map time to x
                const getX = (t) => width - ((now - t) / TIME_WINDOW) * width;
                
                // 2. Draw ASR Regions (Orange Blocks)
                ctx.fillStyle = "rgba(255, 165, 0, 0.3)"; // Orange transparent
                
                // Completed intervals
                for (let evt of asrEvents) {
                    if (now - evt.end > TIME_WINDOW) continue; // Skip old
                    let x1 = getX(evt.start);
                    let x2 = getX(evt.end);
                    if (x1 < 0) x1 = 0;
                    if (x2 > width) x2 = width;
                    ctx.fillRect(x1, 0, x2 - x1, height);
                }
                // Ongoing interval
                if (currentAsrStart !== null) {
                    let x1 = getX(currentAsrStart);
                    if (x1 < 0) x1 = 0;
                    ctx.fillRect(x1, 0, width - x1, height);
                }
                
                // Prune old ASR events
                while(asrEvents.length > 0 && now - asrEvents[0].end > TIME_WINDOW + 1000) {
                    asrEvents.shift();
                }

                // 3. Draw VAD Curve (Blue Line)
                if (vadData.length > 1) {
                    ctx.strokeStyle = "#2563eb"; // Blue
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    
                    let started = false;
                    for (let i = 0; i < vadData.length; i++) {
                        const pt = vadData[i];
                        if (now - pt.t > TIME_WINDOW) continue;
                        
                        const x = getX(pt.t);
                        const y = height - (pt.v * height); // 0 at bottom, 1 at top
                        
                        if (!started) {
                            ctx.moveTo(x, y);
                            started = true;
                        } else {
                            ctx.lineTo(x, y);
                        }
                    }
                    ctx.stroke();
                }
                
                // 4. Draw Threshold Line (Dotted)
                const threshY = height - (0.5 * height);
                ctx.strokeStyle = "#ef4444"; // Red
                ctx.setLineDash([5, 5]);
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(0, threshY);
                ctx.lineTo(width, threshY);
                ctx.stroke();
                ctx.setLineDash([]);

                requestAnimationFrame(drawLoop);
            }
        </script>
    </body>
</html>
"""

# Global Queue for ASR Thread -> WebSocket
msg_queue = queue.Queue(maxsize=1000)
asr_ready_event = threading.Event()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()
app = FastAPI()

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep connection open
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- ASR & VAD Logic ---

class AudioStream:
    def __init__(self, queue_out, file_path=None):
        self.queue_out = queue_out
        self.file_path = file_path
        self.running = False
        self.stream = None
        self.thread = None
        
        # Load VAD model
        print("Loading Silero VAD...")
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                             model='silero_vad',
                                             force_reload=False,
                                             onnx=False)
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.vad_model.to(self.device)
        print(f"VAD running on {self.device}")

        if not self.file_path and not PYAUDIO_AVAILABLE:
             print("Error: pyaudio not installed and no file provided.")
             sys.exit(1)
             
        if not self.file_path:
            self.p = pyaudio.PyAudio()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()

    def _run_loop(self):
        if self.file_path:
            self._process_loop(self._file_generator())
        else:
            self.stream = self.p.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=RATE,
                                    input=True,
                                    frames_per_buffer=CHUNKSZ)
            print("Listening from microphone...")
            self._process_loop(self._mic_generator())

    def _mic_generator(self):
        while self.running:
            try:
                data = self.stream.read(CHUNKSZ, exception_on_overflow=False)
                yield np.frombuffer(data, dtype=np.int16)
            except Exception as e:
                print(f"Mic error: {e}")
                break

    def _file_generator(self):
        print(f"Simulating from {self.file_path}")
        wav, sr = torchaudio.load(self.file_path)
        if sr != RATE:
            wav = torchaudio.transforms.Resample(sr, RATE)(wav)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        
        wav = wav.squeeze(0).numpy()
        wav_int16 = (wav * 32767).astype(np.int16)
        
        step = CHUNKSZ
        delay = CHUNKSZ / RATE
        
        for i in range(0, len(wav_int16), step):
            if not self.running: break
            chunk = wav_int16[i:i+step]
            if len(chunk) < step:
                chunk = np.pad(chunk, (0, step - len(chunk)))
            yield chunk
            time.sleep(delay)

    def _process_loop(self, generator):
        speech_buffer = []
        pre_roll = []
        PRE_ROLL_SIZE = 3
        silence_counter = 0
        is_speech_active = False
        speech_chunks_since_update = 0
        BATCH_SIZE = 30
        MAX_SPEECH_CHUNKS = int(6 * RATE / CHUNKSZ) # 6 seconds
        
        chunks_per_sec = RATE / CHUNKSZ
        silence_chunks_thresh = int(SILENCE_DURATION_MS * chunks_per_sec / 1000)
        
        for audio_int16 in generator:
            if not self.running: break
            
            audio_float32 = torch.from_numpy(audio_int16.astype(np.float32) / 32768.0)
            
            with torch.no_grad():
                prob = self.vad_model(audio_float32.to(self.device), RATE).item()
            
            # Broadcast VAD prob
            try:
                msg_queue.put_nowait({"type": "vad", "prob": float(prob)})
            except:
                pass

            if prob > VAD_THRESHOLD:
                if not is_speech_active:
                    is_speech_active = True
                    # Add pre-roll
                    speech_buffer.extend(pre_roll)
                
                silence_counter = 0
                speech_buffer.append(audio_int16)
                speech_chunks_since_update += 1
                
                if len(speech_buffer) >= MAX_SPEECH_CHUNKS:
                    full_audio = np.concatenate(speech_buffer)
                    self.queue_out.put(("final", full_audio))
                    speech_buffer = []
                    speech_chunks_since_update = 0
                elif speech_chunks_since_update >= BATCH_SIZE:
                    full_audio = np.concatenate(speech_buffer)
                    self.queue_out.put(("partial", full_audio))
                    speech_chunks_since_update = 0
            else:
                if is_speech_active:
                    speech_buffer.append(audio_int16)
                    silence_counter += 1
                    speech_chunks_since_update += 1
                    
                    if silence_counter >= silence_chunks_thresh or len(speech_buffer) >= MAX_SPEECH_CHUNKS:
                        is_speech_active = False
                        full_audio = np.concatenate(speech_buffer)
                        duration_ms = (len(full_audio) / RATE) * 1000
                        
                        if duration_ms >= MIN_SPEECH_MS:
                            self.queue_out.put(("final", full_audio))
                        
                        speech_buffer = []
                        silence_counter = 0
                        speech_chunks_since_update = 0
                    elif speech_chunks_since_update >= BATCH_SIZE:
                        full_audio = np.concatenate(speech_buffer)
                        self.queue_out.put(("partial", full_audio))
                        speech_chunks_since_update = 0
                else:
                    # Keep track of pre-roll chunks while silent
                    pre_roll.append(audio_int16)
                    if len(pre_roll) > PRE_ROLL_SIZE:
                        pre_roll.pop(0)

def asr_worker(audio_queue):
    # Initialize Model
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"Loading ASR model on {device}...")
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        device=device,
        hub="ms",
        fp16=True,
    )
    print("ASR Model Loaded.")
    asr_ready_event.set()

    while True:
        try:
            msg_type, audio_data = audio_queue.get()
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32) / 32768.0)
            
            # Notify start
            msg_queue.put({"type": "asr_start"})
            
            res = model.generate(
                input=[audio_tensor],
                cache={},
                #hotwords=["小 P。"], # KEEP
                batch_size=1,
                #language="中文",
                itn=True,
                disable_pbar=True,
                max_length=50,
            )
            text = res[0]["text"]
            
            # Notify end (with result)
            if text:
                msg_queue.put({"type": msg_type, "text": text})
            else:
                # If no text (empty), still need to close the interval
                 msg_queue.put({"type": msg_type, "text": ""})
                
        except Exception as e:
            print(f"ASR Error: {e}")

# Bridge between Queue and WebSocket
async def broadcast_worker():
    while True:
        try:
            # Send chunks of messages to avoid clogging
            count = 0
            while not msg_queue.empty() and count < 20:
                data = msg_queue.get()
                await manager.broadcast(json.dumps(data))
                count += 1
            await asyncio.sleep(0.02)
        except Exception as e:
            print(f"Broadcast error: {e}")
            await asyncio.sleep(1)

# Main Application Setup
audio_queue = queue.Queue(maxsize=50)

# Parse Args for File Mode
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, help="Path to audio file for simulation")
args, unknown = parser.parse_known_args()

# Start ASR Thread
asr_thread = threading.Thread(target=asr_worker, args=(audio_queue,), daemon=True)
asr_thread.start()

# Start Audio Stream
stream = AudioStream(audio_queue, file_path=args.file)
stream.start()

# Start Broadcast Loop in Background
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_worker())

@app.on_event("shutdown")
def shutdown_event():
    stream.stop()

if __name__ == "__main__":
    import uvicorn
    # Wait for ASR model to load before starting server
    print("Waiting for ASR model to initialize...")
    asr_ready_event.wait()
    print("ASR Ready. Starting Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
