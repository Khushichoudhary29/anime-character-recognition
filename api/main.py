from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = FastAPI()

IMG_SIZE = (128, 128)
MODEL_PATH = "model/saved/best_model.keras"
CLASS_PATH = "model/saved/class_names.json"

templates = Jinja2Templates(directory="frontend")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        name="index.html",
        request=request
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[-3:][::-1]

    top_predictions = [
        {
            "character": class_names[idx],
            "confidence": round(float(predictions[idx] * 100), 2)
        }
        for idx in top_indices
    ]

    return {"top_predictions": top_predictions}