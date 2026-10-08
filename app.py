import io

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from tensorflow import keras

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

app = FastAPI(title="Fashion MNIST API (TensorFlow)")
model = keras.models.load_model("model.keras")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), invert: bool = False):
    """Upload an image. Fashion-MNIST items are light on a dark background;
    pass invert=true for normal dark-on-white photos."""
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("L")
    except Exception:
        raise HTTPException(400, "Invalid image")

    if invert:
        img = ImageOps.invert(img)
    img = img.resize((28, 28))

    x = np.array(img, dtype="float32")[None, ..., None]  # (1, 28, 28, 1)
    probs = model.predict(x, verbose=0)[0]

    top = int(np.argmax(probs))
    return {
        "class": CLASSES[top],
        "confidence": round(float(probs[top]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)