import kagglehub
import os
import shutil
import random

# ===============================
# CONFIGURATION
# ===============================
os.environ["KAGGLEHUB_CACHE"] = "D:/kaggle_cache"

TARGET_BASE = "data/raw"
IMAGES_PER_CHARACTER = 100  
CLEAR_OLD_DATA = True        

# Add more characters here
CHARACTERS = [
    "Naruto Uzumaki",
    "Sasuke Uchiha",
    "Sakura Haruno",
    "Kakashi Hatake",
    "Monkey D. Luffy",
    "Goku",
    "Saitama",
    "Hinata Hyuga",

    # NEW CHARACTERS
    "Itachi Uchiha",
    "Gaara",
    "Vegeta",
    "Ichigo Kurosaki",
    "Levi Ackerman",
    "Mikasa Ackerman",
    "Eren Yeager",
    "Tanjiro Kamado",
    "Nezuko Kamado",
    "Gojo Satoru",
    "Yuji Itadori"
]

# ===============================
# DOWNLOAD DATASET
# ===============================
print("Accessing Kaggle dataset...")
dataset_path = kagglehub.dataset_download(
    "anuragraj03/anime-face-dataset"
)

print(f"Dataset path: {dataset_path}")

os.makedirs(TARGET_BASE, exist_ok=True)

# ===============================
# COPY IMAGES
# ===============================
for character in CHARACTERS:
    source_dir = os.path.join(dataset_path, character)
    target_dir = os.path.join(TARGET_BASE, character)

    if not os.path.exists(source_dir):
        print(f"❌ Character folder not found: {character}")
        continue

    if CLEAR_OLD_DATA and os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    os.makedirs(target_dir, exist_ok=True)

    image_files = [
        file_name for file_name in os.listdir(source_dir)
        if file_name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(image_files)

    selected_images = image_files[:IMAGES_PER_CHARACTER]

    for file_name in selected_images:
        shutil.copy(
            os.path.join(source_dir, file_name),
            os.path.join(target_dir, file_name)
        )

    print(f"✅ {character}: {len(selected_images)} images copied")

print("\nDataset preparation completed successfully.")
print(f"Saved at: {os.path.abspath(TARGET_BASE)}")