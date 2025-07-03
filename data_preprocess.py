from pathlib import Path
import os
import face_recognition
import cv2 as cv
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed


# this module basically does the preprocessing of your dataset
# should only be run once while initial training / splitting data

BASE_PATH = Path(r'C:\Users\Lenovo\Downloads\Image_dataset')  # change to the path of your dataset
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')

# label and image dict
names = [folder.name for folder in BASE_PATH.iterdir() if folder.is_dir()]
label_dict = {name: idx for idx, name in enumerate(names)}
reverse_label_dict = {idx: name for name, idx in label_dict.items()}

image_dict = {}
for image_path in BASE_PATH.rglob('*'):
    if image_path.suffix.lower() in IMAGE_EXTENSIONS:
        label = image_path.parent.name
        image_dict.setdefault(label, []).append(str(image_path))

# data augmentation
def augment_image(image):
    augmented_images = []

    # Horizontal flip
    augmented_images.append(cv.flip(image, 1))

    # Brightness/contrast
    augmented_images.append(cv.convertScaleAbs(image, alpha=1.2, beta=30))
    augmented_images.append(cv.convertScaleAbs(image, alpha=0.8, beta=-30))

    # Gaussian blur
    augmented_images.append(cv.GaussianBlur(image, (5, 5), 0))

    # Rotation ±10°
    h, w = image.shape[:2]
    for angle in [-10, 10]:
        M = cv.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        rotated = cv.warpAffine(image, M, (w, h))
        augmented_images.append(rotated)

    return augmented_images

# face processing function (parallel processing)
def process_image(args):
    path, label_idx = args
    results = []
    try:
        img = cv.imread(path)
        if img is None:
            return None
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        resized_image = cv.resize(rgb_img, (160, 160))

        variants = [resized_image] + augment_image(resized_image)

        for variant in variants:
            locations = face_recognition.face_locations(variant, model="cnn")
            if not locations:
                continue
            encoding = face_recognition.face_encodings(variant, known_face_locations=locations)
            if encoding:
                results.append((encoding[0], label_idx))

        return results if results else None

    except Exception as e:
        print(f"[ERROR] Failed to process {path}: {e}")
        return None

if __name__ == "__main__":
    print("[INFO] Running multi-core CNN-based preprocessing with augmentation...")
    tasks = []
    for label, image_paths in image_dict.items():
        label_idx = label_dict[label]
        for path in image_paths:
            tasks.append((path, label_idx))

    X, y = [], []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            result = future.result()
            if result:
                for encoding, label in result:
                    X.append(encoding)
                    y.append(label)

    print(f"[INFO] Processed {len(X)} face encodings including augmentations.")

    # Save data
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump({'X': X, 'y': y, 'label_map': reverse_label_dict}, f)

    print("[INFO] Saved processed data to processed_data.pkl")
