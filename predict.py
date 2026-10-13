"""Classify one PCB defect crop (or a folder of them).

Usage:
    python predict.py path/to/crop.jpg
    python predict.py path/to/folder_of_crops
"""

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--weights", default="runs/pcb_cls/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=128)
    args = ap.parse_args()

    model = YOLO(args.weights)
    for r in model.predict(args.source, imgsz=args.imgsz, device="cpu", verbose=False, stream=True):
        top5 = ", ".join(f"{model.names[i]} {c:.2f}" for i, c in zip(r.probs.top5, r.probs.top5conf.tolist()))
        print(f"{r.path}\n  -> {model.names[r.probs.top1]} ({r.probs.top1conf:.2%})   [{top5}]")


if __name__ == "__main__":
    main()
