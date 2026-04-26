# Anime Character Recognition System

A mini machine learning web application that predicts anime characters from uploaded images using deep learning.

---

## 📌 Project Overview

The aim of this project is to recognize anime characters from an uploaded image.

The user uploads an image through the web interface, and the system predicts the character name along with the confidence score.

Currently, the model is trained on selected anime characters using a subset of images.

---

## ✨ Features

- Upload anime character image
- Predict top 3 matching characters
- Show confidence percentage
- User-friendly web interface
- Image preview before prediction
- FastAPI backend integration
- Trained using TensorFlow and MobileNetV2

---

## 🎯 Characters Supported

The model currently supports prediction for the following characters:

- Naruto Uzumaki
- Sasuke Uchiha
- Sakura Haruno
- Kakashi Hatake
- Monkey D. Luffy
- Goku
- Saitama
- Hinata Hyuga

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

### Step 1: Open the folder

```bash
cd anime-character-recognition
```

### Step 2: Create virtual environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the application

```bash
uvicorn api.main:app --reload
```

### Step 5: Open in browser

```text
http://127.0.0.1:8000
```

---

## 🧠 Model Information

The model is built using **MobileNetV2 transfer learning**.

- Pretrained on ImageNet
- Fine-tuned for anime face classification
- Softmax output layer for multi-class prediction

---

## 📊 Current Performance

- Model works well on trained dataset with moderate-to-good accuracy
- Performs better on clear, front-facing anime character images
- Accuracy drops for unseen characters or different art styles
- Faster inference on CPU for small batch image inputs
- Limited real-time testing implemented (image-based only for now)
- Performance depends heavily on dataset quality and size
---

## 🔮 Future Improvements

- Increase dataset size for better accuracy and generalization
- Try advanced models like EfficientNet or Vision Transformers
- Enable real-time detection using webcam or video input
- Deploy as a web app using Flask or FastAPI
- Optimize model for faster performance
- Add confidence score filtering for predictions
- Expand support for more anime characters and styles
- Show character details after prediction (name, anime info)
---