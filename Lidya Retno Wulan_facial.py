# Dashboard Streamlit Facial Emotion Recognition
from pathlib import Path
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "emotion_cnn.keras"

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🙂", layout="wide")
st.title("Facial Emotion Recognition menggunakan CNN")
st.write("Aplikasi untuk memprediksi ekspresi emosi dari gambar wajah.")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(OUTPUT_DIR / "class_names.json", encoding="utf-8") as f:
        classes = json.load(f)
    with open(OUTPUT_DIR / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    return model, classes, config

if not MODEL_PATH.exists():
    st.error("Model belum ditemukan. Jalankan `python train.py` atau notebook sampai selesai.")
    st.stop()

model, class_names, config = load_artifacts()
img_size = int(config.get("img_size", 48))

# Ringkasan hasil eksperimen
metrics_path = OUTPUT_DIR / "metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cols = st.columns(4)
    for col, label, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1-Score"],
                               ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]):
        col.metric(label, f"{metrics.get(key, 0):.2%}")

left, right = st.columns([1, 1])
with left:
    st.subheader("Prediksi ekspresi wajah")
    uploaded = st.file_uploader("Upload gambar wajah", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded:
        image = Image.open(uploaded).convert("L")
        st.image(image, caption="Gambar yang diunggah", width=300)
        # Model dilatih dengan gambar wajah grayscale yang sudah ter-crop/terpusat.
        resized = image.resize((img_size, img_size))
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = arr[np.newaxis, ..., np.newaxis]
        probs = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(probs))
        st.success(f"Prediksi: **{class_names[idx].capitalize()}**")
        st.write(f"Tingkat keyakinan model: **{probs[idx]:.2%}**")
        st.subheader("Probabilitas setiap kelas")
        st.bar_chart({name: float(prob) for name, prob in zip(class_names, probs)})

with right:
    st.subheader("Informasi dataset dan hasil pelatihan")
    if (OUTPUT_DIR / "class_distribution.png").exists():
        st.image(str(OUTPUT_DIR / "class_distribution.png"), caption="Distribusi gambar per kelas")
    if (OUTPUT_DIR / "training_curves.png").exists():
        st.image(str(OUTPUT_DIR / "training_curves.png"), caption="Grafik accuracy dan loss")
    if (OUTPUT_DIR / "confusion_matrix.png").exists():
        st.image(str(OUTPUT_DIR / "confusion_matrix.png"), caption="Confusion matrix")
    if (OUTPUT_DIR / "classification_report.csv").exists():
        st.dataframe(__import__("pandas").read_csv(OUTPUT_DIR / "classification_report.csv"),
                     use_container_width=True)

st.caption("Catatan: untuk hasil yang lebih baik, unggah foto wajah yang jelas, menghadap kamera, dan wajah sudah memenuhi sebagian besar area gambar. Prediksi model bukan penilaian pasti atas perasaan seseorang.")
