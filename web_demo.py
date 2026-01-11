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
SILENCE_DURATION_MS = 800
MIN_SPEECH_MS = 200

# HTML Template
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>FunASR Real-time Demo</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }
            #container { background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            #status { text-align: center; color: #666; margin-bottom: 20px; }
            .status-connected { color: green !important; }
            .status-disconnected { color: red !important; }
            #results { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 20px; border-radius: 4px; background: #fafafa; }
            .sentence { margin-bottom: 8px; padding: 5px; border-bottom: 1px solid #eee; }
            .timestamp { color: #888; font-size: 0.8em; margin-right: 10px; }
            #partial-container { height: 60px; border: 2px solid #007bff; padding: 10px; border-radius: 4px; display: flex; align-items: center; background: #f0f7ff; }
            #partial { font-size: 1.2em; color: #0056b3; width: 100%; }
            .placeholder { color: #ccc; font-style: italic; }
        </style>
    </head>
    <body>
        <div id="container">
            <h1>FunASR Live Stream</h1>
            <div id="status" class="status-disconnected">Disconnected</div>
            
            <h3>Finalized Results:</h3>
            <div id="results"></div>
            
            <h3>Current Speech:</h3>
            <div id="partial-container">
                <span id="partial" class="placeholder">Listening...</span>
            </div>
        </div>

        <script>
            var ws = new WebSocket("ws://" + location.host + "/ws");
            var resultsDiv = document.getElementById("results");
            var partialSpan = document.getElementById("partial");
            var statusDiv = document.getElementById("status");

            ws.onopen = function() {
                statusDiv.innerText = "Connected";
                statusDiv.className = "status-connected";
            };

            ws.onmessage = function(event) {
                var data = JSON.parse(event.data);
                
                if (data.type === "partial") {
                    partialSpan.className = "";
                    partialSpan.innerText = data.text;
                } else if (data.type === "final") {
                    partialSpan.innerText = ""; // Clear partial
                    
                    var div = document.createElement("div");
                    div.className = "sentence";
                    
                    var timeSpan = document.createElement("span");
                    timeSpan.className = "timestamp";
                    timeSpan.innerText = new Date().toLocaleTimeString();
                    
                    var textSpan = document.createElement("span");
                    textSpan.innerText = data.text;
                    
                    div.appendChild(timeSpan);
                    div.appendChild(textSpan);
                    
                    resultsDiv.appendChild(div);
                    resultsDiv.scrollTop = resultsDiv.scrollHeight;
                }
            };

            ws.onclose = function() {
                statusDiv.innerText = "Disconnected (Server Stopped)";
                statusDiv.className = "status-disconnected";
            };
        </script>
    </body>
</html>
"""

# Global Queue for ASR Thread -> WebSocket
msg_queue = queue.Queue()

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

# --- ASR & VAD Logic (Adapted from demo_vad.py) ---

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
        silence_counter = 0
        is_speech_active = False
        
        chunks_per_sec = RATE / CHUNKSZ
        silence_chunks_thresh = int(SILENCE_DURATION_MS * chunks_per_sec / 1000)
        
        for audio_int16 in generator:
            if not self.running: break
            
            audio_float32 = torch.from_numpy(audio_int16.astype(np.float32) / 32768.0)
            
            with torch.no_grad():
                prob = self.vad_model(audio_float32.to(self.device), RATE).item()
            
            if prob > VAD_THRESHOLD:
                if not is_speech_active:
                    is_speech_active = True
                
                silence_counter = 0
                speech_buffer.append(audio_int16)
                
                full_audio = np.concatenate(speech_buffer)
                self.queue_out.put(("partial", full_audio))
            else:
                if is_speech_active:
                    speech_buffer.append(audio_int16)
                    silence_counter += 1
                    
                    if silence_counter >= silence_chunks_thresh:
                        is_speech_active = False
                        full_audio = np.concatenate(speech_buffer)
                        duration_ms = (len(full_audio) / RATE) * 1000
                        
                        if duration_ms >= MIN_SPEECH_MS:
                            self.queue_out.put(("final", full_audio))
                        
                        speech_buffer = []
                        silence_counter = 0
                    else:
                        full_audio = np.concatenate(speech_buffer)
                        self.queue_out.put(("partial", full_audio))

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
        hub="ms"
    )
    print("ASR Model Loaded.")

    while True:
        try:
            msg_type, audio_data = audio_queue.get()
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32) / 32768.0)
            
            res = model.generate(
                input=[audio_tensor],
                cache={},
                batch_size=1,
                language="中文",
                itn=True,
                disable_pbar=True,
            )
            text = res[0]["text"]
            
            # Send to global queue for WebSocket broadcasting
            if text:
                msg_queue.put({"type": msg_type, "text": text})
                
        except Exception as e:
            print(f"ASR Error: {e}")

# Bridge between Queue and WebSocket
async def broadcast_worker():
    while True:
        try:
            # Non-blocking get with async sleep to allow other tasks
            while not msg_queue.empty():
                data = msg_queue.get()
                await manager.broadcast(json.dumps(data))
            await asyncio.sleep(0.05)
        except Exception as e:
            print(f"Broadcast error: {e}")
            await asyncio.sleep(1)

# Main Application Setup
audio_queue = queue.Queue()

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
