from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(
    title="Brain Tumor Classifier API",
    description="Upload an image to classify brain tumor type.",
    version="1.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc"      # ReDoc alternative UI
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("model/mobile_brain_model.keras")
class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

def preprocess_image(data):
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = preprocess_image(contents)
    preds = model.predict(image)[0]
    pred_index = np.argmax(preds)
    confidence = float(preds[pred_index]) * 100
    predicted_class = class_names[pred_index]
    return JSONResponse(content={
        "class": predicted_class,
        "confidence": round(confidence, 4)
    })
