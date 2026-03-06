from paddleocr import PaddleOCR
import glob
import logging

# Suppress debug logs from PaddleOCR
logging.getLogger("ppocr").setLevel(logging.WARNING)

print("Loading PaddleOCR...")
# use_angle_cls=True is good for rotated text, lang='japan' for Japanese
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

frames = sorted(glob.glob("samples/frames/*.jpg"))

print(f"Testing on {len(frames)} frames...")
for frame in frames:
    print(f"\n--- {frame} ---")
    results = ocr.ocr(frame, cls=True)
    
    if results and results[0]:
        for line in results[0]:
            text = line[1][0]
            prob = line[1][1]
            if prob > 0.3:
                print(f"[{prob:.2f}] {text}")
    else:
        print("No text detected.")
