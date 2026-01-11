import torch

try:
    print("Loading Silero VAD...")
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=False)
    print("Silero VAD loaded successfully.")
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    model.to(device)
    print(f"Model moved to {device}")

except Exception as e:
    print(f"Error: {e}")
