import os
import shutil
import random
from pathlib import Path

# Configuration parameters
SOURCE_IMAGE_DIR = "./rawdata/images"  # Directory of raw images
SOURCE_LABEL_DIR = "./rawdata/labels"  # Directory of raw labels
DEST_DIR = "./dataset"                 # Destination directory (train/val/test will be created inside)
SPLIT_RATIOS = (0.7, 0.2, 0.1)         # Split ratios: train:val:test
RANDOM_SEED = 42                       # Random seed to ensure reproducibility

# Supported image file extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]

def main():
    # Create target directory structure
    subsets = ["train", "val", "test"]
    for subset in subsets:
        Path(f"{DEST_DIR}/{subset}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{DEST_DIR}/{subset}/labels").mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = [
        f for f in os.listdir(SOURCE_IMAGE_DIR)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]
    
    if not image_files:
        print(f"Error: No image files found in {SOURCE_IMAGE_DIR}")
        return
    
    # Shuffle file order
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)
    
    # Calculate the number of files in each subset
    total = len(image_files)
    train_count = int(total * SPLIT_RATIOS[0])
    val_count = int(total * SPLIT_RATIOS[1])
    # Remaining files are assigned to the test set
    
    # Split files into subsets
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count+val_count]
    test_files = image_files[train_count+val_count:]
    
    print(f"Dataset split results:")
    print(f"Total files: {total}")
    print(f"Training set: {len(train_files)} files")
    print(f"Validation set: {len(val_files)} files")
    print(f"Test set: {len(test_files)} files")
    
    # Function for copying files
    def copy_files(file_list, subset):
        for img_file in file_list:
            # Copy image file
            src_img = os.path.join(SOURCE_IMAGE_DIR, img_file)
            dst_img = os.path.join(DEST_DIR, subset, "images", img_file)
            shutil.copy2(src_img, dst_img)
            
            # Copy corresponding label file (assumes label file has same name with .txt extension)
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label = os.path.join(SOURCE_LABEL_DIR, label_file)
            
            if os.path.exists(src_label):
                dst_label = os.path.join(DEST_DIR, subset, "labels", label_file)
                shutil.copy2(src_label, dst_label)
            else:
                print(f"Warning: Label file {label_file} not found for image {img_file}")
    
    # Copy files to each subset
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")
    
    print("Dataset splitting completed!")
    print(f"New directory structure created under {DEST_DIR}")

if __name__ == "__main__":
    main()