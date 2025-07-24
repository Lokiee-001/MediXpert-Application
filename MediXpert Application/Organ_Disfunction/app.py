from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

app = Flask(__name__)

# Path to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your models
BRAIN_MODEL_PATH = 'brain_tumor_detection_model.keras'
brain_model = load_model(BRAIN_MODEL_PATH)

PNEUMONIA_MODEL_PATH = 'pneumonia_detection_model_v2.keras'
pneumonia_model = load_model(PNEUMONIA_MODEL_PATH)

# Route for homepage (upload form)
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle brain tumor detection
@app.route('/upload_brain', methods=['POST'])
def upload_brain_file():
    if 'file' not in request.files:
        return "No file part in the request", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)  # Save the file to the uploads folder

        # Process the uploaded image
        img = load_img(file_path, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the brain tumor model
        prediction = brain_model.predict(img_array)[0][0]
        result_message = "The image is affected by a brain tumor." if prediction > 0.5 else "The image is not affected by a brain tumor."

        # Pass result and image to the result page
        return render_template('result.html', message=result_message, image_url=file_path)

# Route to handle pneumonia detection
@app.route('/upload_lungs', methods=['POST'])
def upload_lungs_file():
    if 'file' not in request.files:
        return "No file part in the request", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)  # Save the file to the uploads folder

        # Process the uploaded image
        img = load_img(file_path, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the pneumonia model
        prediction = pneumonia_model.predict(img_array)[0][0]
        result_message = "The X-Ray indicates pneumonia." if prediction > 0.5 else "The X-Ray does not indicate pneumonia."

        # Pass result and image to the result page
        return render_template('result.html', message=result_message, image_url=file_path)

# Main block to run the Flask app
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)  # Create uploads folder if not exists
    app.run(debug=True)
