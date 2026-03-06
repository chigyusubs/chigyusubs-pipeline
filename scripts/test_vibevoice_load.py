import torch
from vibevoice.asr.model import VibeVoiceASR

def test():
    print("Loading scerz/VibeVoice-ASR-4bit...")
    model = VibeVoiceASR.from_pretrained("scerz/VibeVoice-ASR-4bit")
    print("Model loaded successfully!")
    print(f"Model device: {model.device if hasattr(model, 'device') else 'unknown'}")

if __name__ == "__main__":
    test()
