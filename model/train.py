# model/train.py

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Fix Windows CPU crash issue

import tensorflow as tf
import json
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 4
EPOCHS = 5
DATA_DIR = "data/raw"
MODEL_DIR = "model/saved"

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# =========================
# LOAD DATASET
# =========================
print("\nLoading dataset...\n")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print(f"\nClasses found: {class_names}")

# Save class names
with open(f"{MODEL_DIR}/class_names.json", "w") as f:
    json.dump(class_names, f)

# =========================
# PREPROCESSING
# =========================
normalization = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda images, labels: (normalization(images), labels)
)

val_ds = val_ds.map(
    lambda images, labels: (normalization(images), labels)
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# =========================
# MODEL
# =========================
print("\nBuilding model...\n")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# CALLBACKS
# =========================
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"{MODEL_DIR}/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

# =========================
# TRAINING
# =========================
print("\nStarting training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# =========================
# SAVE FINAL MODEL
# =========================
model.save(f"{MODEL_DIR}/anime_model.keras")

print("\nModel saved successfully.")

# =========================
# PLOT ACCURACY
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.grid(True)

plot_path = f"{MODEL_DIR}/accuracy_plot.png"
plt.savefig(plot_path)

print(f"Accuracy plot saved at: {plot_path}")