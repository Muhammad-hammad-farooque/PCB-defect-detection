"""Export the trained classifier to ONNX and copy it into the Streamlit app.

Usage:
    python export_onnx.py --weights runs/pcb_cls/weights/best.pt
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/pcb_cls/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=128)
    args = ap.parse_args()

    onnx_path = YOLO(args.weights).export(format="onnx", imgsz=args.imgsz, simplify=True)
    dst = Path("app/model/pcb_cls.onnx")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, dst)
    print(f"Copied {onnx_path} -> {dst}")


if __name__ == "__main__":
    main()
