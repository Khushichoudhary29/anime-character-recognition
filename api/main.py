from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = FastAPI()

templates = Jinja2Templates(directory="frontend")

model = tf.keras.models.load_model("model/saved/best_model.keras")

with open("model/saved/class_names.json", "r") as f:
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
    image = image.resize((128, 128))

    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    top_indices = np.argsort(predictions)[-3:][::-1]

    top_predictions = []

    for idx in top_indices:
        top_predictions.append({
            "character": class_names[idx],
            "confidence": round(float(predictions[idx] * 100), 2)
        })

    return {
        "top_predictions": top_predictions
    }