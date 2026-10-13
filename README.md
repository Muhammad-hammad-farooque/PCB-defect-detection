# PCB Defect Classifier — YOLO11n-cls

A lightweight deep-learning classifier that identifies **6 types of printed circuit board (PCB) defects**,
trained with Ultralytics **YOLO11n-cls** (nano) on the Kaggle
[PCB Defects dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects), and served through a
**Streamlit** web app.

| Defect class | Description |
|---|---|
| `missing_hole` | A drilled hole that should be there is absent |
| `mouse_bite` | Small notches eaten into the edge of a copper trace or pad |
| `open_circuit` | A break in a trace that should be continuous |
| `short` | Unwanted copper connecting two traces that should be separate |
| `spur` | A thin, unwanted copper protrusion sticking out of a trace |
| `spurious_copper` | Leftover copper in a place where there should be none |

**Test accuracy: 99.5%** (439 / 441 held-out defect images).

---

## Table of contents
- [How it works](#how-it-works)
- [Results](#results)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Web app](#web-app)
- [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

---

## How it works

The Kaggle dataset is a **detection** dataset: 693 full-size PCB photographs (~3000×2000 px) with
Pascal VOC XML bounding boxes around each defect. Because this project trains a **classifier**, the
pipeline first turns each annotated defect into its own small image:

```
Full PCB photo + XML boxes
        │  prepare_dataset.py
        ▼
Square crop around each defect (2× the box size, for surrounding context)
        │  split 70 / 15 / 15 by source photo → data/pcb_cls/{train,val,test}/<class>/
        ▼
YOLO11n-cls (ImageNet-pretrained) fine-tuned at 128×128 px      ← train.py
        │
        ▼
best.pt ──► evaluate.py (test metrics)
        └─► export_onnx.py ──► app/model/pcb_cls.onnx ──► Streamlit app (onnxruntime)
```

Design choices:
- **Split by source photo, not by crop.** One photo can contain several defects. Keeping all of a
  photo's crops in the same split stops near-duplicates from leaking between train and test, so the
  test score is honest.
- **Rotation folder skipped.** The dataset's `rotation/` folder contains pre-rotated copies of the same
  images. It is excluded, and orientation variety comes from training augmentation instead
  (horizontal + vertical flips, ±90° rotation), since a PCB defect looks the same at any orientation.
- **ONNX for deployment.** The web app runs the exported ONNX model with `onnxruntime`, so it doesn't
  need PyTorch or Ultralytics. This keeps the deployed app small enough for Streamlit's free tier.

---

## Results

### Dataset after cropping (2,953 defect images)

| Class | Train | Val | Test |
|---|---:|---:|---:|
| missing_hole | 349 | 67 | 81 |
| mouse_bite | 341 | 71 | 80 |
| open_circuit | 350 | 68 | 64 |
| short | 349 | 70 | 72 |
| spur | 348 | 69 | 71 |
| spurious_copper | 363 | 67 | 73 |
| **Total** | **2,100** | **412** | **441** |

### Training

| Setting | Value |
|---|---|
| Model | `yolo11n-cls.pt` (ImageNet-pretrained, ~1.5M parameters) |
| Image size | 128 × 128 |
| Epochs | 30 (early-stopping patience 10) |
| Batch size | 32 |
| Augmentation | flip LR 0.5, flip UD 0.5, rotation ±90° (plus Ultralytics defaults) |
| Hardware | CPU only (4 logical cores) |
| Training time | ~26 minutes (~52 s / epoch) |
| Best validation top-1 accuracy | 99.5% |

### Test set (441 images never seen during training)

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| missing_hole | 1.000 | 1.000 | 1.000 |
| mouse_bite | 1.000 | 1.000 | 1.000 |
| open_circuit | 1.000 | 1.000 | 1.000 |
| short | 0.986 | 1.000 | 0.993 |
| spur | 1.000 | 0.986 | 0.993 |
| spurious_copper | 0.986 | 0.986 | 0.986 |
| **Accuracy** | | | **0.995** |

Confusion matrix (rows = true class, columns = predicted class):

| | missing_hole | mouse_bite | open_circuit | short | spur | spurious_copper |
|---|---:|---:|---:|---:|---:|---:|
| **missing_hole** | 81 | 0 | 0 | 0 | 0 | 0 |
| **mouse_bite** | 0 | 80 | 0 | 0 | 0 | 0 |
| **open_circuit** | 0 | 0 | 64 | 0 | 0 | 0 |
| **short** | 0 | 0 | 0 | 72 | 0 | 0 |
| **spur** | 0 | 0 | 0 | 0 | 70 | 1 |
| **spurious_copper** | 0 | 0 | 0 | 1 | 0 | 72 |

The exported ONNX model, run through the web app's own preprocessing code, scores the same 439/441
on the test set.

---

## Project structure

```
pcb detection/
├── app/
│   ├── streamlit_app.py      # Web UI: upload → (optional) crop → classify
│   ├── requirements.txt      # Lightweight deps for deployment (no torch)
│   └── model/
│       └── pcb_cls.onnx      # Exported model used by the app (committed)
├── prepare_dataset.py        # VOC XML → cropped class-folder dataset
├── train.py                  # Fine-tune YOLO11n-cls
├── evaluate.py               # Per-class metrics + confusion matrix on the test split
├── predict.py                # Command-line inference on images or folders
├── export_onnx.py            # best.pt → ONNX, copied into app/model/
├── requirements.txt          # Full dev/training dependencies
├── .gitignore
├── data/                     # (git-ignored) raw dataset + generated crops
└── runs/                     # (git-ignored) training outputs: weights, curves, logs
```

---

## Setup

Requirements: **Python 3.11** and [uv](https://docs.astral.sh/uv/). A GPU is not needed.

```powershell
# 1. Create the virtual environment
uv venv .venv --python 3.11

# 2. Install CPU-only PyTorch (much smaller than the CUDA build)
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install everything else (Ultralytics, OpenCV, Streamlit, onnxruntime, ...)
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

> Have an NVIDIA GPU? Install the CUDA build of PyTorch in step 2 instead, then train with
> `--device 0`.

On macOS / Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python` throughout.

---

## Usage

All commands run from the project root.

### 1. Get the dataset
Download the zip from [Kaggle: akhatova/pcb-defects](https://www.kaggle.com/datasets/akhatova/pcb-defects)
and extract it so the layout is:

```
data/raw/PCB_DATASET/
├── Annotations/<Class>/*.xml
├── images/<Class>/*.jpg
└── rotation/ ...            (ignored)
```

Any location works. Pass it with `--src`.

### 2. Build the classification dataset
```powershell
.\.venv\Scripts\python prepare_dataset.py --src data/raw/PCB_DATASET
```
| Option | Default | Meaning |
|---|---|---|
| `--out` | `data/pcb_cls` | Output folder |
| `--context` | `2.0` | Crop side = defect box side × context |
| `--min-size` | `64` | Minimum crop side in pixels |
| `--val` / `--test` | `0.15` / `0.15` | Split fractions |
| `--seed` | `42` | Random seed for the split |

### 3. Train
```powershell
.\.venv\Scripts\python train.py
```
| Option | Default |
|---|---|
| `--epochs` | `30` |
| `--imgsz` | `128` |
| `--batch` | `32` |
| `--device` | `cpu` |
| `--workers` | `2` (use `0` on low-memory machines) |

The best weights are saved to `runs/pcb_cls/weights/best.pt`. Training curves and sample batches are
saved in the same run folder.

### 4. Evaluate
```powershell
.\.venv\Scripts\python evaluate.py
```
Prints precision / recall / F1 for each class, plus the confusion matrix on the test split.

### 5. Predict from the command line
```powershell
.\.venv\Scripts\python predict.py path\to\defect_crop.jpg
.\.venv\Scripts\python predict.py path\to\folder_of_crops
```
Example output:
```
data\pcb_cls\test\short\01_short_16_0.jpg
  -> short (100.00%)   [short 1.00, spurious_copper 0.00, spur 0.00, missing_hole 0.00, open_circuit 0.00]
```

### 6. Export for the web app
```powershell
.\.venv\Scripts\python export_onnx.py
```
Writes `runs/pcb_cls/weights/best.onnx` and copies it to `app/model/pcb_cls.onnx`.

---

## Web app

```powershell
.\.venv\Scripts\streamlit run app/streamlit_app.py
```
Then open http://localhost:8501.

1. **Upload** a PCB image (JPG / PNG / BMP).
2. For a **full board photo**, keep *Select the defect region* on and drag the square over the defect.
   This toggle is on by default for images larger than 400 px. For an **already-cropped close-up**,
   switch it off.
3. The app shows the **predicted defect type**, its **confidence**, and a bar chart of the
   probabilities for all six classes.

The model runs on the server with `onnxruntime`, and the browser only displays the interface.
Pre-cropped samples to try are in `data/pcb_cls/test/<class>/` after step 2.

---

## Deploying to Streamlit Community Cloud

1. Push the project to a **public GitHub repository**. `.gitignore` already excludes `.venv/`, `data/`,
   `runs/`, the raw Kaggle archive and `*.pt` files, but keeps `app/model/pcb_cls.onnx` (~6 MB),
   which the app needs.
   ```powershell
   git init
   git add .
   git commit -m "PCB defect classifier with Streamlit app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **Create app**.
3. Select the repository and branch, and set **Main file path** to `app/streamlit_app.py`.
4. Click **Deploy**. Streamlit Cloud installs `app/requirements.txt` automatically because it is
   next to the main file, so PyTorch and Ultralytics are not installed on the server.

---

## Limitations

- **This is a classifier, not a detector.** It labels a patch that already contains one defect. It
  doesn't find defects on a full board by itself, which is why the app asks you to box the defect.
  Finding defects automatically would need a YOLO11n **detection** model trained on the same boxes.
- **The 99.5% is measured on crops made from the dataset's own boxes.** In the app, a loose or
  off-centre box can reduce accuracy. Keep the defect roughly centred, with a little surrounding copper.
- **There is no "no defect" class.** Every input gets one of the six labels, even a clean patch of board.
- **All images come from one dataset** (PKU PCB images with synthetic defects, same lighting and
  boards). Expect lower accuracy on photos from other cameras or board types until you fine-tune on them.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Install or training gets killed / the PC freezes | Close other programs, and train with `--workers 0 --batch 16` |
| `Model not found at app/model/pcb_cls.onnx` | Run `export_onnx.py` after training |
| `No VOC .xml annotations found` | Point `--src` at the folder that contains `Annotations/` and `images/` |
| Weights saved under `runs/classify/...` | Older copies of `train.py` used a relative project path. The current version saves to `runs/pcb_cls/` |
| `ImportError: libGL.so.1` on Streamlit Cloud | Make sure the app's requirements don't include `opencv-python`. The app only needs `app/requirements.txt` |

---

## Acknowledgements

- Dataset: [PCB Defects](https://www.kaggle.com/datasets/akhatova/pcb-defects) on Kaggle, from the
  Open Lab on Human Robot Interaction, Peking University.
- Model and training framework: [Ultralytics YOLO11](https://docs.ultralytics.com/).
- Web framework: [Streamlit](https://streamlit.io/). Cropping widget:
  [streamlit-cropper](https://github.com/turner-anderson/streamlit-cropper).
