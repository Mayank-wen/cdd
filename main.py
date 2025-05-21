from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load your model once when the server starts
model = tf.keras.models.load_model("crop_disease_model_final.keras")

# Define the input image size your model expects
IMG_SIZE = (224, 224)

# Mapping class indices to your provided class names
CLASS_NAMES = {
    0: "Apple Black rot",
    1: "Apple Healthy",
    2: "Apple Scab",
    3: "Bell pepper Bacterial spot",
    4: "Bell pepper Healthy",
    5: "Cedar apple rust",
    6: "Citrus Black spot",
    7: "Citrus Healthy",
    8: "Citrus canker",
    9: "Citrus greening",
    10: "Corn Common rust",
    11: "Corn Gray leaf spot",
    12: "Corn Healthy",
    13: "Corn Northern Leaf Blight",
    14: "Grape Black Measles",
    15: "Grape Black rot",
    16: "Grape Healthy",
    17: "Grape Isariopsis Leaf Spot",
    18: "Peach Bacterial spot",
    19: "Peach Healthy",
    20: "Potato Early blight",
    21: "Potato Healthy",
    22: "Potato Late blight",
    23: "Tomato Bacterial spot",
    24: "Tomato Early blight",
    25: "Tomato Healthy",
    26: "Tomato Late blight",
    27: "Tomato Leaf Mold",
    28: "Tomato Mosaic virus",
    29: "Tomato Septoria leaf spot",
    30: "Tomato Spider mites",
    31: "Tomato Target Spot",
    32: "Tomato Yellow Leaf Curl Virus"
}

@app.route('/')
def home():
    return "Crop Disease Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        img = image.load_img(file, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        preds = model.predict(img_array)
        pred_class_idx = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))

        pred_class_name = CLASS_NAMES.get(pred_class_idx, "Unknown")

        return jsonify({
            'predicted_class_index': int(pred_class_idx),
            'predicted_class_name': pred_class_name,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
