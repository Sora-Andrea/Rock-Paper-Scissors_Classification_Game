import os
import random
import shutil

# Config
SOURCE_DIR = "rps_dataset"
DEST_DIR = "rps_split_dataset"
CLASSES = ["rock", "paper", "scissors"]
TRAIN_RATIO = 0.75
SEED = 42

random.seed(SEED)

# Directory structure
for split in ["train", "val"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

# Split files based on class
for cls in CLASSES:
    class_dir = os.path.join(SOURCE_DIR, cls)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(".png")]

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Copy training images
    for img in train_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(DEST_DIR, "train", cls, img)
        shutil.copy2(src, dst)

    # Copy validation images
    for img in val_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(DEST_DIR, "val", cls, img)
        shutil.copy2(src, dst)

    print(f"{cls}: {len(train_images)} train, {len(val_images)} val")

print(f"\nDataset split complete:{DEST_DIR}")
