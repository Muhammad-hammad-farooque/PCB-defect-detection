"""Streamlit front end for the YOLO11n-cls PCB defect classifier.

Runs the exported ONNX model with onnxruntime, so the deployed app doesn't need torch/ultralytics.

Local run:
    streamlit run app/streamlit_app.py
"""

import ast
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

MODEL_PATH = Path(__file__).parent / "model" / "pcb_cls.onnx"


@st.cache_resource
def load_model():
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map  # Ultralytics stores class names and imgsz here
    names = ast.literal_eval(meta["names"])
    imgsz = ast.literal_eval(meta["imgsz"])[0]
    return session, names, imgsz


def preprocess(img: Image.Image, size: int) -> np.ndarray:
    """Match Ultralytics classify inference transforms: resize short side, center crop, scale to [0, 1], NCHW."""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    x = np.asarray(img, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1)[None]


def classify(img: Image.Image):
    session, names, imgsz = load_model()
    probs = session.run(None, {session.get_inputs()[0].name: preprocess(img, imgsz)})[0][0]
    return {names[i].replace("_", " "): float(p) for i, p in enumerate(probs)}


st.set_page_config(page_title="PCB Defect Classifier", page_icon="🔍", layout="centered")
st.title("PCB Defect Classifier")
st.caption("YOLO11n-cls trained on the Kaggle PCB defects dataset: missing hole, mouse bite, "
           "open circuit, short, spur, spurious copper.")

if not MODEL_PATH.exists():
    st.error(f"Model not found at `{MODEL_PATH}`. Train the model and run `python export_onnx.py` first.")
    st.stop()

uploaded = st.file_uploader("Upload a PCB image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded is None:
    st.info("Upload a close-up of a defect, or a full board photo and draw a box around the defect.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
use_crop = st.toggle("Select the defect region", value=max(image.size) > 400,
                     help="The model was trained on small patches around a single defect. "
                          "For full board photos, draw a box around the defect.")

if use_crop:
    st.write("Drag the box over the defect:")
    patch = st_cropper(image, realtime_update=True, box_color="#FF4B4B", aspect_ratio=(1, 1))
else:
    patch = image

col1, col2 = st.columns([1, 1.4])
with col1:
    st.image(patch, caption="Model input", use_container_width=True)
with col2:
    scores = classify(patch)
    label, conf = max(scores.items(), key=lambda kv: kv[1])
    st.metric("Prediction", label.title(), f"{conf:.1%} confidence", delta_color="off")
    df = pd.DataFrame({"defect": list(scores), "probability": list(scores.values())})
    st.bar_chart(df.sort_values("probability", ascending=False), x="defect", y="probability", horizontal=True)
