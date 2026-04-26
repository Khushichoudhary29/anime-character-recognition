import kagglehub
import os
import shutil

# 1. Setup Cache on D: Drive
os.environ["KAGGLEHUB_CACHE"] = "D:/kaggle_cache"

# 2. Download/Locate Dataset
print("Accessing dataset...")
path = kagglehub.dataset_download("anuragraj03/anime-face-dataset")

# 3. List of exact character folder names
# I've updated these to match the Kaggle folder names exactly
characters = [
    "Naruto Uzumaki",  
    "Sasuke Uchiha", 
    "Sakura Haruno", 
    "Kakashi Hatake",
    "Monkey D. Luffy",
    "Goku",
    "Saitama"
]

target_base = "data/raw"
num_images = 20  # Your requested limit

for char in characters:
    source_dir = os.path.join(path, char)
    dest_dir = os.path.join(target_base, char)
    
    # Check if folder exists
    if os.path.exists(source_dir):
        os.makedirs(dest_dir, exist_ok=True)
        
        # Get all images and take only the first 20
        files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        subset = files[:num_images]
        
        for f in subset:
            shutil.copy(os.path.join(source_dir, f), os.path.join(dest_dir, f))
            
        print(f"✅ Copied {len(subset)} images for {char}")
    else:
        print(f"❌ Folder '{char}' not found. Check spelling!")

print(f"\nDone! Your dataset is ready at: {os.path.abspath(target_base)}")