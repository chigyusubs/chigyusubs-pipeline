from faster_whisper import WhisperModel
import time

print("Loading faster-whisper large-v3 model on ROCm...")
start = time.time()
# Loading with int8 is the fastest way to test if the CTranslate2 HIP backend works
model = WhisperModel("large-v3", device="cuda", compute_type="int8")
print(f"Model loaded in {time.time() - start:.2f} seconds.")

print("\nRunning a short 1-minute transcription test with VAD...")
segments, info = model.transcribe(
    "samples/WEDNESDAY_DOWNTOWN_2024-01-24 _#363.mp4",
    language="ja",
    vad_filter=True
)

# Only grab the first 5 segments to prove it works
for i, segment in enumerate(segments):
    print(f"[{segment.start:.2f} -> {segment.end:.2f}] {segment.text}")
    if i >= 4:
        break
        
print("Test successful!")
