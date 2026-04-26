# Anime Character Recognition System 🎌

A mini machine learning web application that predicts anime characters from uploaded images using deep learning.

This project is built using **TensorFlow, FastAPI, HTML, CSS, and JavaScript**.

---

## 📌 Project Overview

The aim of this project is to recognize anime characters from an uploaded image.

The user uploads an image through the web interface, and the system predicts the character name along with the confidence score.

Currently, the model is trained on selected anime characters using a subset of images.

---

## ✨ Features

- Upload anime character image
- Predict character name
- Show confidence percentage
- User-friendly web interface
- FastAPI backend integration
- Deep learning model using MobileNetV2

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

> Note: Due to limited dataset size, predictions may sometimes vary for unseen images.

---

## 🔮 Future Improvements

- Increase dataset size
- Better prediction accuracy
- More anime characters
- Better UI design

---