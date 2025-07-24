
# 🧠 MediXpert - AI-Based Medical Diagnostic System

**MediXpert** is a human-friendly, AI-powered medical diagnostic web application that assists in the **detection of brain tumors, pneumonia, and bone fractures** using deep learning models. It simplifies the diagnostic process by providing fast, accurate, and interpretable predictions from medical images.

---

## 📌 About the Project

MediXpert is built with the vision of supporting early-stage disease detection using Artificial Intelligence. By leveraging **Convolutional Neural Networks (CNNs)**, the system processes medical scans (X-rays/MRIs) and outputs disease classification along with suggestions and model confidence.

This system was developed over several iterations of model training and tuning. The repository may contain **supporting files, model versions, experiments, or archived datasets** that were part of the development process.

---

## 🔬 AI & CNN-Based Architecture

Each diagnostic module is powered by a separate CNN model:

- Multiple convolutional layers extract spatial features (edges, patterns).
- Pooling layers reduce dimensionality while preserving features.
- Dropout and regularization layers improve generalization.
- Final Dense layers produce classification outputs using Softmax/Sigmoid.

The models are deployed with **Flask** to serve real-time predictions in a web-friendly interface.

---

## 🧪 Detection Modules & Training Summary

| 🧠 Module             | 📊 Dataset Size | 🎯 Accuracy | ⚙️ CNN Details              | 📁 Train/Test Split |
|----------------------|------------------|-------------|-----------------------------|----------------------|
| Brain Tumor          | 20000 images    | ~96%        | 4 Conv layers, ReLU, Dropout | 80:20                |
| Pneumonia Detection  | 17000 images    | ~94%        | 3 Conv layers, MaxPooling    | 80:20                |
| Bone Fracture        | 15000 images   | ~93%        | 5 Conv layers, BatchNorm     | 75:25                |

All models were trained using:
- **TensorFlow/Keras**
- **Data Augmentation**
- **Early Stopping**
- **Image normalization (grayscale, resizing)**

---

## ⚙️ Features

- 📤 Upload X-ray or MRI scans
- 🧠 Real-time predictions using AI
- 📈 Confidence score with clear labels
- 💬 Dynamic suggestions & precautions
- 🌐 Simple and clean web interface
- 🗂️ Modular architecture for scalability

---

## 🖼️ Web Application Screenshot

![MediXpert Web Application Demo](Web%20Application%20Demo%20Pic.png)

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, Bootstrap
- **Backend:** Python (Flask)
- **AI Models:** Convolutional Neural Networks (CNN)
- **Libraries:** TensorFlow, Keras, OpenCV, NumPy, Matplotlib

---

🚀 How to Run MediXpert Locally


1️⃣ Clone the Repository




git clone https://github.com/your-username/MediXpert.git
cd MediXpert



2️⃣ Install Dependencies




pip install -r requirements.txt



3️⃣ Run the Flask Application




python app.py



4️⃣ Open in Your Browser



http://127.0.0.1:5000



📁 Project Structure



MediXpert/


├── app.py                       # Flask main application


├── model_brain.h5               # Trained model for brain tumor


├── model_pneumonia.h5           # Trained model for pneumonia


├── model_bone.h5                # Trained model for bone fracture


├── static/                      # CSS, JS, and image assets


├── templates/                   # HTML templates (Flask views)


├── uploads/                     # Uploaded images directory


├── requirements.txt             # Python dependencies


└── Web Application Demo Pic.png # Demo screenshot used in README



👨‍💻 Developed By


Lokeshwar L (B.Tech CSE)


🔗 LinkedIn Profile [LinkedIn](https://www.linkedin.com/in/lam-lokeshwar-730bb12a0/)



💡 Passionate about healthcare, AI, and impactful problem solving.




📢 Notes & Credits


This project was developed as a research-based diagnostic assistant.

Several variations of models and file versions may exist in the repository from past testing and iterations.

Datasets are sourced from trusted public datasets:

Brain Tumor Dataset (Kaggle)

Chest X-ray Dataset (Kaggle)

Bone Dataset (Kaggle)

📜 License
This project is licensed under the MIT License.
You are free to use, modify, and share with attribution.


-------------------------------------------------------------------------------

***NOTE***
“In machine learning, nothing is truly wasted — every failed attempt teaches the model to succeed better.” ***so It consists of some extra practiced files***

 🙏 Thank You

Thank you for exploring MediXpert!
This project reflects my passion for combining healthcare and artificial intelligence to make diagnostic tools more accessible and intelligent.

If you find this useful or inspiring, feel free to ⭐ star the repository, suggest improvements,  Connect with me on [LinkedIn](https://www.linkedin.com/in/lam-lokeshwar-730bb12a0/)
.

                                                                   
                                                         
