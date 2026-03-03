import easyocr
import glob

print("Loading EasyOCR...")
reader = easyocr.Reader(['ja', 'en'])

frames = sorted(glob.glob("samples/frames/*.jpg"))

print(f"Testing on {len(frames)} frames...")
for frame in frames:
    print(f"\n--- {frame} ---")
    results = reader.readtext(frame)
    for (bbox, text, prob) in results:
        # Only print if confidence is above 30% to filter noise
        if prob > 0.3:
            print(f"[{prob:.2f}] {text}")
