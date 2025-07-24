from flask import Flask, render_template, request, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
import uuid
import imghdr

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

BRAIN_MODEL_PATH = 'brain_tumor_detection_model.keras'
brain_model = load_model(BRAIN_MODEL_PATH)

PNEUMONIA_MODEL_PATH = 'pneumonia_detection_model_v2.keras'
pneumonia_model = load_model(PNEUMONIA_MODEL_PATH)

BONE_FRACTURE_MODEL_PATH = 'bone_fracture_model.h5'
bone_fracture_model = load_model(BONE_FRACTURE_MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_type(file_stream):
    image_type = imghdr.what(file_stream)
    file_stream.seek(0)
    return image_type in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

def get_dynamic_content(damage_percentage, condition):
    # Dynamic recommendation and precaution logic based on damage percentage
    thresholds = [(10, 0), (20, 1), (30, 2), (40, 3), (50, 4), (60, 5), (70, 6), (80, 7), (90, 8), (100, 9)]
    index = next((i for t, i in thresholds if damage_percentage <= t), 9)

    content_map = {
        "brain": [
            ["Monitor symptoms; minimal concern.", "Annual MRI advised."],
            ["Consult neurologist within 3 months.", "Repeat scan in 6 months."],
            ["Neurological evaluation recommended.", "Discuss early intervention."],
            ["Medical follow-up necessary.", "Check for subtle symptoms."],
            ["Detailed MRI and consultation.", "Discuss biopsy possibility."],
            ["Advanced imaging advised.", "Begin treatment discussion."],
            ["Surgical options considered.", "CT + MRI required."],
            ["Hospitalization may be needed.", "Start treatment immediately."],
            ["Chemo/surgery planning.", "Intensive care likely."],
            ["Emergency intervention required.", "Immediate hospitalization."]
        ],
        "lungs": [
            ["Rest and monitor.", "Stay hydrated."],
            ["Consult physician.", "OTC meds may help."],
            ["Chest X-ray follow-up.", "Start antibiotics."],
            ["Pulmonologist consult.", "Start full antibiotics."],
            ["Medical observation.", "Diagnostic tests."],
            ["Possible hospitalization.", "Nebulizer/inhaler may help."],
            ["Begin oxygen therapy.", "Strict medication."],
            ["Hospital stay advised.", "High-dose treatment."],
            ["ICU may be needed.", "Advanced tests required."],
            ["Emergency treatment.", "Intensive care."]
        ],
        "bone": [
            ["Cold compress.", "Minimal concern."],
            ["Doctor visit in 3 days.", "X-ray confirmation."],
            ["Ortho exam required.", "Temporary splint advised."],
            ["Consult orthopedic.", "Plaster suggested."],
            ["Weekly X-ray.", "Pain meds."],
            ["Cast/splint.", "Biweekly imaging."],
            ["Surgery possible.", "Bone density test."],
            ["Bone realignment.", "Surgical fixation."],
            ["Fracture treatment.", "Physiotherapy post-healing."],
            ["Emergency surgery.", "Ambulance transport."]
        ]
    }

    recommendations, precautions = content_map[condition][index]
    return [recommendations], [precautions]

@app.route('/upload_brain', methods=['POST'])
def upload_brain_file():
    if 'file' not in request.files:
        flash('No file part in the request', 'error')
        return render_template('index.html', active_tab='brain')

    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return render_template('index.html', active_tab='brain')

    if file and allowed_file(file.filename):
        if not validate_image_type(file.stream):
            flash('Invalid image type.', 'error')
            return render_template('index.html', active_tab='brain')

        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        try:
            img = load_img(file_path, target_size=(150, 150))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = brain_model.predict(img_array)[0][0]
            is_affected = prediction > 0.5
            accuracy = round(prediction * 100, 2) if is_affected else round((1 - prediction) * 100, 2)
            damage_percentage = round((1 - prediction) * 100, 2) if is_affected else round(prediction * 100, 2)
            health_status = "Affected" if is_affected else "Healthy"

            message = "The MRI scan indicates presence of a brain tumor." if is_affected else "The MRI scan shows no signs of brain tumor."
            recommendations, precautions = get_dynamic_content(damage_percentage, "brain") if is_affected else (["Maintain regular health checkups."], ["Stay safe and healthy."])

            return render_template('index.html', message=message, image_url=file_path,
                                   accuracy=accuracy, damage_percentage=damage_percentage,
                                   health_status=health_status,
                                   recommendations=recommendations, precautions=precautions,
                                   active_tab='brain')
        except Exception as e:
            flash(f'Error processing image: {str(e)}', 'error')
            return render_template('index.html', active_tab='brain')

    flash('Allowed file types are png, jpg, jpeg', 'error')
    return render_template('index.html', active_tab='brain')

@app.route('/upload_lungs', methods=['POST'])
def upload_lungs_file():
    if 'file' not in request.files:
        flash('No file part in the request', 'error')
        return render_template('index.html', active_tab='lungs')

    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return render_template('index.html', active_tab='lungs')

    if file and allowed_file(file.filename):
        if not validate_image_type(file.stream):
            flash('Invalid image type.', 'error')
            return render_template('index.html', active_tab='lungs')

        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        try:
            img = load_img(file_path, target_size=(150, 150))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = pneumonia_model.predict(img_array)[0][0]
            is_affected = prediction > 0.5
            accuracy = round(prediction * 100, 2) if is_affected else round((1 - prediction) * 100, 2)
            damage_percentage = round((1 - prediction) * 100, 2) if is_affected else round(prediction * 100, 2)
            health_status = "Affected" if is_affected else "Healthy"

            message = "The chest X-ray indicates presence of pneumonia." if is_affected else "The chest X-ray shows no signs of pneumonia."
            recommendations, precautions = get_dynamic_content(damage_percentage, "lungs") if is_affected else (["Maintain good respiratory health."], ["Stay hygienic."])

            return render_template('index.html', message=message, image_url=file_path,
                                   accuracy=accuracy, damage_percentage=damage_percentage,
                                   health_status=health_status,
                                   recommendations=recommendations, precautions=precautions,
                                   active_tab='lungs')
        except Exception as e:
            flash(f'Error processing image: {str(e)}', 'error')
            return render_template('index.html', active_tab='lungs')

    flash('Allowed file types are png, jpg, jpeg', 'error')
    return render_template('index.html', active_tab='lungs')

@app.route('/upload_bone', methods=['POST'])
def upload_bone_file():
    if 'file' not in request.files:
        flash('No file part in the request', 'error')
        return render_template('index.html', active_tab='bone')

    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return render_template('index.html', active_tab='bone')

    if file and allowed_file(file.filename):
        if not validate_image_type(file.stream):
            flash('Invalid image type.', 'error')
            return render_template('index.html', active_tab='bone')

        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        try:
            img = load_img(file_path, target_size=(150, 150))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = bone_fracture_model.predict(img_array)[0][0]
            is_affected = prediction > 0.5
            accuracy = round(prediction * 100, 2) if is_affected else round((1 - prediction) * 100, 2)
            damage_percentage = round((1 - prediction) * 100, 2) if is_affected else round(prediction * 100, 2)
            health_status = "Affected" if is_affected else "Healthy"

            message = "The X-ray indicates presence of a bone fracture." if is_affected else "The X-ray shows no signs of bone fracture."
            recommendations, precautions = get_dynamic_content(damage_percentage, "bone") if is_affected else (["Maintain strong bones."], ["Use safety gear."])

            return render_template('index.html', message=message, image_url=file_path,
                                   accuracy=accuracy, damage_percentage=damage_percentage,
                                   health_status=health_status,
                                   recommendations=recommendations, precautions=precautions,
                                   active_tab='bone')
        except Exception as e:
            flash(f'Error processing image: {str(e)}', 'error')
            return render_template('index.html', active_tab='bone')

    flash('Allowed file types are png, jpg, jpeg', 'error')
    return render_template('index.html', active_tab='bone')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=5000, debug=True)
