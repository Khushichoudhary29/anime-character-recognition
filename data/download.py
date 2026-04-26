import kagglehub
import os
import shutil

# Setup cache on D drive
os.environ["KAGGLEHUB_CACHE"] = "D:/kaggle_cache"

print("Accessing dataset...")
path = kagglehub.dataset_download("anuragraj03/anime-face-dataset")

characters = [
    "Naruto Uzumaki",
    "Sasuke Uchiha",
    "Sakura Haruno",
    "Kakashi Hatake",
    "Monkey D. Luffy",
    "Goku",
    "Saitama",
    "Hinata Hyuga"
]

target_base = "data/raw"
num_images = 50

for char in characters:
    source_dir = os.path.join(path, char)
    dest_dir = os.path.join(target_base, char)

    if os.path.exists(source_dir):
        # Create fresh folder
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)

        os.makedirs(dest_dir, exist_ok=True)

        files = [
            f for f in os.listdir(source_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        subset = files[:num_images]

        for file_name in subset:
            shutil.copy(
                os.path.join(source_dir, file_name),
                os.path.join(dest_dir, file_name)
            )

        print(f"✅ Copied {len(subset)} images for {char}")

    else:
        print(f"❌ Folder '{char}' not found")

print(f"\nDataset ready at: {os.path.abspath(target_base)}")