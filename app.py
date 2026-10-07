import io
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps

app = Flask(__name__)
model = tf.keras.models.load_model("fashion_minist_model.keras")

LABELS = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify(error="no file"), 400

    img = Image.open(io.BytesIO(request.files["file"].read())).convert("L")
    if request.form.get("invert") == "true":
        img = ImageOps.invert(img)
    img = img.resize((28, 28))

    x = np.array(img, dtype="float32")[None, ..., None] / 255.0
    p = model.predict(x, verbose=0)[0]
    top = np.argsort(p)[::-1][:3]

    return jsonify(
        prediction=LABELS[top[0]],
        top3=[{"label": LABELS[i], "prob": round(float(p[i]), 4)} for i in top],
    )


if __name__ == "__main__":
    from waitress import serve
    print("Running on http://127.0.0.1:5000")
    serve(app, host="127.0.0.1", port=5000)