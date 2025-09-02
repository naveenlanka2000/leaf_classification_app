import tensorflow as tf
from flask import Flask, request, render_template
import numpy as np
from PIL import Image

app = Flask(__name__)

# Make sure the model path matches your actual file
model_path = "leaf_classification_model.h5"
model = tf.keras.models.load_model(model_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"
    file = request.files["file"]
    if file.filename == "":
        return "No file selected"

    image = Image.open(file.stream).resize((224, 224))  # adjust size as per model
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = np.argmax(prediction, axis=1)[0]

    return f"Prediction: {result}"

if __name__ == "__main__":
    app.run(debug=True)
