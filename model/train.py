import tensorflow as tf
import pathlib, json, os

# Config
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "data/raw"
SAVE_DIR = "model/saved"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load class names based on your folder structure
CLASS_NAMES = sorted([p.name for p in pathlib.Path(DATA_DIR).iterdir() if p.is_dir()])
json.dump(CLASS_NAMES, open(f"{SAVE_DIR}/class_names.json", "w"))

# Load and augment data
def load_ds(subset):
    return tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="categorical", seed=42, validation_split=0.2, subset=subset,
        class_names=CLASS_NAMES)

train_ds = load_ds("training")
val_ds = load_ds("validation")

# Define MobileNetV2 Base
base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base.trainable = False 

# Build Model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.layers.Rescaling(1./255)(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# Compile and Train (Phase 1)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save Final Result
model.save(f"{SAVE_DIR}/best_model.h5")