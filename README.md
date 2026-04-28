# 🎭 Anime Character Recognition System

A machine learning based mini web application that predicts anime characters.
---

## 📌 Project Overview

The aim of this project is to recognize anime characters from an uploaded image.

The user uploads an anime image through the web interface, and the system predicts the **top 3 most likely character matches** along with their confidence percentages.

---

## ✨ Features

- Upload anime character image
- Drag and drop image upload
- Image preview before prediction
- Predict top 3 matching characters
- Show confidence percentage
- Confidence progress bars
- Best match insight card
- Recent prediction history panel
- User-friendly professional dashboard UI
- FastAPI backend integration
- TensorFlow MobileNetV2 model

---

## 🛠️ Tech Stack

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- FastAPI
- Uvicorn

### Machine Learning
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Pillow

---

## 📁 Project Structure

```text
anime-character-recognition/
│
├── api/
│   └── main.py
│
├── frontend/
│   └── index.html
│
├── model/
│   ├── train.py
│   └── saved/
│       ├── best_model.keras
│       ├── class_names.json
│       └── accuracy_plot.png
│
├── data/
│   └── raw/
│
├── requirements.txt
│
└── README.md
```

---

## 🚀 How to Run the Project

### Step 1: Open the project folder

```bash
cd anime-character-recognition
```

---

### Step 2: Create virtual environment

```bash
python -m venv venv
```

---

### Step 3: Activate virtual environment

#### Windows

```bash
venv\Scripts\activate
```

---

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 5: Run the application

```bash
uvicorn api.main:app --reload
```

---

### Step 6: Open in browser

```text
http://127.0.0.1:8000
```

---

## 🧠 Model Information

The model is built using **MobileNetV2 transfer learning**.

### Model Details

- Pretrained on ImageNet
- Transfer learning based classification
- Softmax output layer
- Multi-class prediction
- Top 3 ranked outputs

### Input Processing

- image resize: 128 × 128
- RGB conversion
- normalization
- batch dimension expansion

---

## ⚙️ Working Process

The application works in the following steps:

```text
Upload Image
   ↓
Image Preprocessing
   ↓
Model Prediction
   ↓
Top 3 Confidence Ranking
   ↓
Frontend Dashboard Display
```

---

## 📊 Current Performance

- Good performance on trained dataset
- Better results for clear front-facing anime faces
- Top 3 ranking improves usability
- Fast prediction on CPU
- Suitable for mini project demonstration
- Accuracy depends on dataset size and image quality

---

## 🖥️ Current UI Enhancements

The current interface includes:

- modern dashboard layout
- preview panel
- prediction result panel
- confidence bars
- insight card
- recent history section
- drag-and-drop upload

These upgrades make the project presentation-ready.

---

## 🔮 Future Improvements

- increase dataset size
- support more anime characters
- improve model accuracy
- use EfficientNet / Vision Transformer
- real-time webcam prediction
- character information card
- API-based anime details
- deployment on Render / Railway
- mobile responsive UI

---

## 🎓 Conclusion

This project demonstrates the practical use of **machine learning, transfer learning, and web application integration** for anime character recognition.