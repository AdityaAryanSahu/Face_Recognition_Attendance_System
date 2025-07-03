import os
import cv2
import pickle
import face_recognition
from pathlib import Path
from data_preprocess import process_image
from concurrent.futures import ProcessPoolExecutor, as_completed


# this is where the data of new registered members is processed
# dont run data_preprocess.py again
# use this instead

PROCESSED_DATA_FILE = "processed_data.pkl"
ENCODING_MODEL = "cnn" # can change to "hog" if too slow
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def register_new_student(student_name, save_dir):
    print(f"[INFO] Registering new student: {student_name}")

    # Load existing processed data
    if os.path.exists(PROCESSED_DATA_FILE):
        with open(PROCESSED_DATA_FILE, "rb") as f:
            data = pickle.load(f)
            X = data["X"]
            y = data["y"]
            label_map = data["label_map"]  
    else:
        X, y, label_map = [], [], {}

    # Check if student already exists
    if student_name in label_map.values():
        label_id = [k for k, v in label_map.items() if v == student_name][0]
        print(f"[INFO] Student already exists with label ID: {label_id}")
    else:
        label_id = max(label_map.keys(), default=-1) + 1
        label_map[label_id] = student_name
        print(f"[INFO] Assigned new label ID: {label_id}")

    # Load and process images
    student_path = Path(save_dir)
    if not student_path.exists():
        print(f"[ERROR] No folder found for student at: {student_path}")
        return

    images = list(student_path.glob("**/*"))
    tasks = [(str(img_path), label_id) for img_path in images if img_path.suffix.lower() in IMAGE_EXTENSIONS]

    count = 0
    for task in tasks:
        result = process_image(task)
        if result:
             for encoding, label in result:
                X.append(encoding)
                y.append(label)
                count += 1

    print(f"[INFO] Added {count} encoding(s) for '{student_name}'")

    # Save updated data
    with open(PROCESSED_DATA_FILE, "wb") as f:
        pickle.dump({"X": X, "y": y, "label_map": label_map}, f)

    print("[SUCCESS] Updated processed_data.pkl")
