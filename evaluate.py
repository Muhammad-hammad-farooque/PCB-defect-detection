"""Evaluate the trained classifier on the held-out test split.

Usage:
    python evaluate.py --weights runs/pcb_cls/weights/best.pt
"""

import argparse
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/pcb_cls/weights/best.pt")
    ap.add_argument("--data", default="data/pcb_cls")
    ap.add_argument("--imgsz", type=int, default=128)
    args = ap.parse_args()

    model = YOLO(args.weights)
    names = model.names  # {idx: class_name}
    test_dir = Path(args.data) / "test"

    y_true, y_pred = [], []
    for cls_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        files = sorted(cls_dir.glob("*.jpg"))
        for r in model.predict([str(f) for f in files], imgsz=args.imgsz, device="cpu", verbose=False, stream=True):
            y_true.append(cls_dir.name)
            y_pred.append(names[r.probs.top1])

    labels = sorted(set(y_true))
    print(classification_report(y_true, y_pred, labels=labels, digits=3))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(" " * 18 + "".join(f"{l[:8]:>9}" for l in labels))
    for l, row in zip(labels, confusion_matrix(y_true, y_pred, labels=labels)):
        print(f"{l:<18}" + "".join(f"{v:>9}" for v in row))


if __name__ == "__main__":
    main()
