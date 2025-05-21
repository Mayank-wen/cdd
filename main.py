from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Class names
class_names = [
    "Apple Black rot", "Apple Healthy", "Apple Scab",
    "Bell pepper Bacterial spot", "Bell pepper Healthy",
    "Cedar apple rust", "Citrus Black spot", "Citrus Healthy",
    "Citrus canker", "Citrus greening", "Corn Common rust",
    "Corn Gray leaf spot", "Corn Healthy", "Corn Northern Leaf Blight",
    "Grape Black Measles", "Grape Black rot", "Grape Healthy",
    "Grape Isariopsis Leaf Spot", "Peach Bacterial spot", "Peach Healthy",
    "Potato Early blight", "Potato Healthy", "Potato Late blight",
    "Tomato Bacterial spot", "Tomato Early blight", "Tomato Healthy",
    "Tomato Late blight", "Tomato Leaf Mold", "Tomato Mosaic virus",
    "Tomato Septoria leaf spot", "Tomato Spider mites",
    "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus"
]

# Load model
try:
    model = load_model("crop_disease_model_final.keras")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", e)
    model = None

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})

    try:
        # Read image bytes
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        preds = model.predict(img_array)
        class_index = np.argmax(preds[0])
        confidence = float(np.max(preds[0]))
        class_name = class_names[class_index]

        return {
            "predicted_class": class_name,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
