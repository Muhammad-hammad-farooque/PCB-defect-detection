"""Turn the Kaggle PCB defect dataset (akhatova/pcb-defects) into a YOLO classification dataset.

The source dataset is a *detection* dataset: full PCB photos + Pascal VOC XML boxes.
For YOLO11n-cls we crop every annotated defect (with some surrounding context) and
write it into an ImageFolder layout that Ultralytics expects:

    data/pcb_cls/
        train/<class>/*.jpg
        val/<class>/*.jpg
        test/<class>/*.jpg

The split is done per *source image*, so crops from the same PCB photo never end up
in both train and test (that would leak and inflate accuracy).

Usage:
    python prepare_dataset.py --src data/raw/PCB_DATASET
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_voc(xml_path):
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename")
    boxes = []
    for obj in root.iter("object"):
        name = obj.findtext("name").strip().lower().replace(" ", "_")
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (int(float(bb.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((name, x1, y1, x2, y2))
    return filename, boxes


def square_crop(img, x1, y1, x2, y2, context, min_size):
    """Square crop centred on the box, enlarged by `context` so the classifier sees surrounding copper."""
    h, w = img.shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * context
    side = max(side, min_size)
    half = side / 2
    nx1, ny1 = int(max(0, cx - half)), int(max(0, cy - half))
    nx2, ny2 = int(min(w, cx + half)), int(min(h, cy + half))
    return img[ny1:ny2, nx1:nx2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/PCB_DATASET", help="folder containing Annotations/ and images/")
    ap.add_argument("--out", default="data/pcb_cls")
    ap.add_argument("--context", type=float, default=2.0, help="crop side = box side * context")
    ap.add_argument("--min-size", type=int, default=64, help="minimum crop side in pixels")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    xmls = sorted(p for p in src.rglob("*.xml") if "rotation" not in p.parts)
    if not xmls:
        raise SystemExit(f"No VOC .xml annotations found under {src.resolve()}")

    # Index images by stem so we don't depend on the exact folder layout of the zip.
    images = {p.stem: p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS and "rotation" not in p.parts}

    # Group by the defect class of each source image so the split is stratified.
    by_class = defaultdict(list)
    for x in xmls:
        _, boxes = parse_voc(x)
        if boxes and x.stem in images:
            by_class[boxes[0][0]].append(x)

    rng = random.Random(args.seed)
    split_of = {}
    for cls, files in by_class.items():
        rng.shuffle(files)
        n_test = round(len(files) * args.test)
        n_val = round(len(files) * args.val)
        for i, f in enumerate(files):
            split_of[f] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    if out.exists():
        shutil.rmtree(out)

    counts = Counter()
    for xml_path, split in tqdm(split_of.items(), desc="Cropping"):
        _, boxes = parse_voc(xml_path)
        img = cv2.imread(str(images[xml_path.stem]))
        if img is None:
            continue
        for i, (cls, x1, y1, x2, y2) in enumerate(boxes):
            crop = square_crop(img, x1, y1, x2, y2, args.context, args.min_size)
            if crop.size == 0:
                continue
            d = out / split / cls
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{xml_path.stem}_{i}.jpg"), crop)
            counts[(split, cls)] += 1

    classes = sorted({c for _, c in counts})
    print(f"\n{'class':<18}{'train':>7}{'val':>7}{'test':>7}")
    for c in classes:
        print(f"{c:<18}" + "".join(f"{counts[(s, c)]:>7}" for s in ("train", "val", "test")))
    print(f"\nDataset written to {out.resolve()}")


if __name__ == "__main__":
    main()
