import time
import torch
import numpy as np
import threading
import queue
import sys
import os
import argparse
import torchaudio
import logging

# Suppress warnings and logs
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('funasr').setLevel(logging.ERROR)

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
SILENCE_DURATION_MS = 800  # End sentence after 800ms silence
MIN_SPEECH_MS = 200        # Ignore speech shorter than 200ms

class AudioStream:
    def __init__(self, queue_out):
        if not PYAUDIO_AVAILABLE:
             print("Error: pyaudio is not installed.")
             print("Please install it using: pip install pyaudio")
             print("Note: On macOS, you may need to install portaudio first: brew install portaudio")
             sys.exit(1)
             
        self.p = pyaudio.PyAudio()
        self.queue_out = queue_out
        self.running = False
        self.stream = None
        
        # Load VAD model
        print("Loading Silero VAD...")
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                             model='silero_vad',
                                             force_reload=False,
                                             onnx=False)
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils
        
        # Determine device for VAD
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.vad_model.to(self.device)
        print(f"VAD running on {self.device}")

    def start(self):
        self.running = True
        self.stream = self.p.open(format=pyaudio.paInt16,
                                channels=1,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNKSZ)
        self.thread = threading.Thread(target=self._read_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

    def _read_loop(self):
        print("Listening...")
        self._process_loop(self._mic_generator())

    def _mic_generator(self):
        while self.running:
            try:
                data = self.stream.read(CHUNKSZ, exception_on_overflow=False)
                audio_int16 = np.frombuffer(data, dtype=np.int16)
                yield audio_int16
            except Exception as e:
                print(f"Error reading stream: {e}")
                break

    def _process_loop(self, chunk_generator):
        speech_buffer = []
        silence_counter = 0
        is_speech_active = False
        
        chunks_per_sec = RATE / CHUNKSZ
        silence_chunks_thresh = int(SILENCE_DURATION_MS * chunks_per_sec / 1000)
        
        for audio_int16 in chunk_generator:
            if not self.running: break

            try:
                # Convert to float32 tensor for VAD
                audio_float32 = torch.from_numpy(audio_int16.astype(np.float32) / 32768.0)
                
                # VAD Check
                with torch.no_grad():
                    prob = self.vad_model(audio_float32.to(self.device), RATE).item()
                
                if prob > VAD_THRESHOLD:
                    # Speech detected
                    if not is_speech_active:
                        is_speech_active = True
                        print("\n[Speech Detected] Starting ASR...")
                    
                    silence_counter = 0
                    speech_buffer.append(audio_int16)
                    
                    # Send updated full buffer for "Real-time" update
                    full_audio = np.concatenate(speech_buffer)
                    self.queue_out.put(("partial", full_audio))
                    
                else:
                    # Silence
                    if is_speech_active:
                        speech_buffer.append(audio_int16) # Keep adding trailing silence
                        silence_counter += 1
                        
                        if silence_counter >= silence_chunks_thresh:
                            # Finalize sentence
                            is_speech_active = False
                            full_audio = np.concatenate(speech_buffer)
                            
                            duration_ms = (len(full_audio) / RATE) * 1000
                            if duration_ms >= MIN_SPEECH_MS:
                                self.queue_out.put(("final", full_audio))
                            else:
                                print(f"\n[Ignored] Too short ({int(duration_ms)}ms)")
                            
                            speech_buffer = []
                            silence_counter = 0
                        else:
                            full_audio = np.concatenate(speech_buffer)
                            self.queue_out.put(("partial", full_audio))
            except Exception as e:
                print(f"Error in process loop: {e}")
                break

class FileStream(AudioStream):
    def __init__(self, queue_out, file_path):
        # Skip AudioStream init to avoid pyaudio check/init
        self.queue_out = queue_out
        self.file_path = file_path
        self.running = False
        self.stream = None
        
        # Load VAD model (duplicate code, but cleaner than refactoring AudioStream too much)
        print("Loading Silero VAD...")
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                             model='silero_vad',
                                             force_reload=False,
                                             onnx=False)
        (self.get_speech_timestamps, _, _, _, _) = utils
        
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.vad_model.to(self.device)
        print(f"VAD running on {self.device}")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._read_loop)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _read_loop(self):
        print(f"Simulating stream from {self.file_path}...")
        self._process_loop(self._file_generator())

    def _file_generator(self):
        # Load audio using torchaudio
        wav, sr = torchaudio.load(self.file_path)
        if sr != RATE:
            resampler = torchaudio.transforms.Resample(sr, RATE)
            wav = resampler(wav)
        
        # Mix down to mono
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        wav = wav.squeeze(0).numpy()
        # Convert to int16 for compatibility
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
        
        print("\nFile finished.")
        # self.running = False # Don't stop immediately to allow processing to finish

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to audio file for simulation")
    args = parser.parse_args()

    # Setup ASR Model
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Loading ASR model on {device}...")
    
    # Reusing initialization from demo1.py
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        device=device,
        hub="ms"
    )

    # Queue for communication
    audio_queue = queue.Queue()
    
    # Start Audio Stream
    if args.file:
        stream = FileStream(audio_queue, args.file)
    else:
        stream = AudioStream(audio_queue)
    
    stream.start()
    
    try:
        while True:
            try:
                # Wait for data
                msg_type, audio_data = audio_queue.get(timeout=0.1)
                
                audio_tensor = torch.from_numpy(audio_data.astype(np.float32) / 32768.0)
                
                # Using parameters from demo1.py
                res = model.generate(
                    input=[audio_tensor],
                    cache={},
                    batch_size=1,
                    # hotwords=["开放时间"], # Optional
                    language="中文", # Default language
                    itn=True,
                    disable_pbar=True,
                )                
                text = res[0]["text"]
                
                if msg_type == "partial":
                    sys.stdout.write("\033[K") 
                    print(f"\rPartial: {text}", end="", flush=True)
                elif msg_type == "final":
                    sys.stdout.write("\033[K")
                    print(f"\rFinal: {text}")

            except queue.Empty:
                if args.file and not stream.thread.is_alive():
                    break
                continue
                
    except KeyboardInterrupt:
        print("\nStopping...")
        stream.stop()

if __name__ == "__main__":
    main()
