"""Flask application for classifying plant leaves.

The application exposes a web interface where users can upload an image of a
leaf.  The image is processed by a TensorFlow model which returns the predicted
class along with a confidence score.  Results are rendered in a simple HTML
template.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# Load the trained model once at start‑up.  The file shipped with the exercise
# uses the ``.keras`` format, so make sure we point to the correct path.
MODEL_PATH = "leaf_classifier_model.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)

# Optional mapping from numeric class index to human readable label.  Adjust
# these names to match the classes used when training the model.
CLASS_NAMES = ["Class 0", "Class 1", "Class 2", "Class 3"]


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalise the uploaded image for the model.

    Parameters
    ----------
    image:
        A Pillow image instance uploaded by the user.

    Returns
    -------
    numpy.ndarray
        A ``(1, 224, 224, 3)`` float32 array with values in the range
        ``[0, 1]`` ready for prediction.
    """

    image = image.convert("RGB")  # ensure three colour channels
    image = image.resize((224, 224))
    img_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


@app.route("/")
def index() -> str:
    """Render the upload form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict() -> str:
    """Handle image uploads and display prediction results."""

    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    image = Image.open(file.stream)
    img_array = preprocess_image(image)
    prediction = MODEL.predict(img_array)
    predicted_index = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]))

    label = (
        CLASS_NAMES[predicted_index]
        if predicted_index < len(CLASS_NAMES)
        else str(predicted_index)
    )

    return render_template(
        "result.html",
        label=label,
        confidence=f"{confidence:.2%}",
    )


if __name__ == "__main__":  # pragma: no cover - manual run
    app.run(debug=True)
