# app.py
import os
from pathlib import Path

# --- (Optional) quiet TensorFlow logs BEFORE importing TF ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")        # hide info/warn
# os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")     # disable oneDNN msgs/ops

import tensorflow as tf
from flask import Flask, request, render_template
import numpy as np
from PIL import Image

# ---------------------------
# Model File (robust absolute path)
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "leaf_classifier_model.keras"   # adjust if you move it

if not MODEL_PATH.exists():
    print("Model file not found at", MODEL_PATH)

def load_tf_model(path: Path):
    # TF 2.15 / Keras 2.x: compile=False avoids optimizer rebuild on load
    return tf.keras.models.load_model(str(path), compile=False)

# Try to load the model; fail gracefully
model = None
try:
    if MODEL_PATH.exists():
        print("Loading model from:", MODEL_PATH)
        model = load_tf_model(MODEL_PATH)
        print("Model loaded.")
    else:
        print("Skipping load: file missing.")
except Exception as e:
    print("Model load failed:", e)

model_name = MODEL_PATH.name

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)

@app.route("/")
def index():
    # Ensure you have templates/index.html in: <project>/templates/index.html
    return render_template("index.html")

@app.route("/health")
def health():
    return {
        "tensorflow": tf.__version__,
        "model_loaded": model is not None,
        "model_name": model_name,
        "model_path": str(MODEL_PATH),
    }, 200

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return f"Model not loaded. Expected at: {MODEL_PATH}", 500

    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if not file.filename:
        return "No file selected", 400

    # ---- Image preprocessing ----
    # If you trained with MobileNetV2 preprocessing, uncomment the two lines marked (MNv2)
    image = Image.open(file.stream).convert("RGB").resize((224, 224))

    # (Default) simple rescaling:
    arr = (np.array(image, dtype=np.float32) / 255.0)[None, ...]  # shape: (1, 224, 224, 3)

    # (MNv2) use this instead of the line above if you trained with MobileNetV2 preprocessing:
    # from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    # arr = preprocess_input(np.array(image, dtype=np.float32))[None, ...]

    # ---- Inference ----
    preds = model.predict(arr)
    class_id = int(np.argmax(preds, axis=1)[0])

    # If you have a class-name list, map here:
    # classes = ["classA", "classB", "classC", ...]
    # return {"prediction": classes[class_id], "class_id": class_id}, 200

    return {"prediction": class_id}, 200

if __name__ == "__main__":
    # On Windows, the reloader can execute code twice; disable if it causes issues
    app.run(debug=True, use_reloader=False)
