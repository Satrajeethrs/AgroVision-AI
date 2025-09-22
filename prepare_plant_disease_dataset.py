import os
import shutil
import random
from pathlib import Path

# Paths
DATASET_DIR = 'Validation_Set'
OUTPUT_DIR = 'plant_disease_dataset'
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

def prepare_dataset():
    if os.path.exists(OUTPUT_DIR):
        print(f"{OUTPUT_DIR} already exists. Remove it to re-run.")
        return
    os.makedirs(OUTPUT_DIR)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

    for class_name in os.listdir(DATASET_DIR):
        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        images = [f for f in os.listdir(class_path) if f.lower().endswith('.jpg')]
        random.shuffle(images)
        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]
        for split, split_imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            split_dir = os.path.join(OUTPUT_DIR, split, class_name)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_imgs:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copy2(src, dst)
    print(f"Dataset prepared in {OUTPUT_DIR}/ with train/val/test splits.")

if __name__ == "__main__":
    prepare_dataset()
