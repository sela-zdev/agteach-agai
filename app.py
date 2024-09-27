from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
model = tf.keras.models.load_model('./model/plant_disease_cnn_x.keras')

class_list = []
# Load from the JSON file
with open('./metadata/disease_list.json', 'r') as json_file:
    class_list = json.load(json_file)

def get_disease_detail(img_path):
    try:
        # Load the image
        img = image.load_img(img_path, target_size=(150, 150))
    except Exception as e:
        return {"error": "Failed to load image", "details": str(e)}

    try:
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Predict the class
        pred_class = model.predict(img_array)
        pred_class = np.argmax(pred_class, axis=1)
        class_idx = int(pred_class[0])
        
        return class_list[class_idx]
    except IndexError as e:
        return {"error": "Prediction returned an invalid index", "details": str(e)}
    except Exception as e:
        return {"error": "An error occurred during prediction", "details": str(e)}

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    image_file = request.files['image']
    
    # Ensure the file is an image
    if not image_file or image_file.filename == '':
        return jsonify({'error': 'No selected image'}), 400
    
    try:
        # Use io.BytesIO to read the image file into a PIL Image
        img_file = io.BytesIO(image_file.read())
        
        # Get the predicted disease detail
        predicted_class = get_disease_detail(img_file)

        # Check if predicted_class is an error message
        if isinstance(predicted_class, dict) and 'error' in predicted_class:
            return jsonify(predicted_class), 500  # Internal Server Error

        return jsonify({'data': predicted_class})
    
    except Exception as e:
        return jsonify({'error': 'An error occurred while processing the image', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)