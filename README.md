# Face Recognition Attendance System

A real-time Face Recognition–based Attendance System built with **OpenCV**, **face_recognition**, **PyQt6**, and **scikit-learn**. It detects and recognizes faces through a webcam and marks attendance. If the face is unknown, it prompts the user to register, captures face images, retrains the model, and updates the dataset automatically.

---

## Features

-  Real-time face recognition using a Voting Classifier 
   (SVM, KNN, RandomForest, DecisionTree, Naive Bayes, Logistic Regression )
-  Live webcam feed with PyQt6 UI
-  Automatic attendance logging (`attendance.csv`)
-  Detects unknown faces and prompts user registration
-  Captures 25 face images during registration
-  Automatic model retraining after registration
-  Dark mode interface with modern layout

---

## Tech Stack

- Python 3.10
- OpenCV
- dlib & face_recognition
- scikit-learn (VotingClassifier, StandardScaler)
- PyQt6 (GUI)
- concurrent.futures (for parallel image processing)

---

## How It Works

1. The system loads all face encodings and trained model.
2. It continuously reads from the webcam and detects faces.
3. If recognized:
   - Attendance is logged.
   - A welcome message is shown.
4. If unrecognized:
   - After a few consistent "Unknown" frames, the right panel appears.
   - User enters their name and clicks "Register".
   - 50 face images are captured.
   - Data is processed and model is retrained.
   - The system is automatically updated.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/AdityaAryanSahu/Face_Recognition_Attendance_System.git
cd Face_Recognition_Attendance_System
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# or
source venv/bin/activate   # On macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

>  Make sure you're using **Python 3.10** for `dlib` compatibility.

### 4. Run the Application

```bash
python ui.py
```

---

## Contributed By

**Aditya Aryan Sahu**  
🔗 [GitHub](https://github.com/AdityaAryanSahu)

---
