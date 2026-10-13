"""Train YOLO11n-cls on the cropped PCB defect dataset (CPU-friendly defaults).

Usage:
    python train.py                 # defaults below
    python train.py --epochs 50 --imgsz 160
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pcb_cls")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    model = YOLO("yolo11n-cls.pt")  # ImageNet-pretrained nano classifier, downloaded on first run
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=10,
        project=str(Path("runs").resolve()),  # absolute, or newer Ultralytics nests it under runs/classify/
        name="pcb_cls",
        exist_ok=True,
        # PCB defects look the same flipped or rotated, so use more geometric augmentation than the defaults.
        fliplr=0.5,
        flipud=0.5,
        degrees=90,
        seed=42,
    )
    print("Best weights: runs/pcb_cls/weights/best.pt")


if __name__ == "__main__":
    main()
