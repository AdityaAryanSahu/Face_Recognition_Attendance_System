import sys
import cv2
import numpy as np
import face_recognition
import csv
import datetime
import os
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
                             QHBoxLayout, QLineEdit, QMessageBox)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from collect_faces import capture_images
from registration_new import register_new_student
from model_retrain import retrain_model 
from data_resplit import resplit_data 
from collections import deque, Counter
from PyQt6.QtWidgets import QDialog, QFormLayout


os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" # forced turned off warning


# Load face data
import pickle
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)
    known_encodings = data['X']
    known_labels = data['y']
    label_map = data['label_map']
    reverse_map = {v: k for k, v in label_map.items()}

# Your preloaded model parts

with open("voting_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
with open("selector.pkl", "rb") as f:
        selector = pickle.load(f)
with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)


prediction_buffer = deque(maxlen=5)

class FaceAttendanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Attendance System")
        self.setGeometry(200, 100, 800, 600)
        main_layout = QHBoxLayout(self)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)

        self.message_label = QLabel("Initializing camera...", self)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dark_stylesheet = """
        QWidget {
        background-color: #121212;
        color: #eeeeee;
        font-family: Arial;
        font-size: 18px;
        }
        QLabel {
        color: #ffffff;
        }
        QPushButton {
        background-color: #2e2e2e;
        color: #ffffff;
        border: 1px solid #444;
        padding: 6px 12px;
        border-radius: 6px;
        }
        QPushButton:hover {
        background-color: #444444;
        }
        QLineEdit {
        background-color: #1e1e1e;
        color: #ffffff;
        border: 1px solid #555;
        padding: 4px;
        border-radius: 4px;
        }
        """

        app.setStyleSheet(dark_stylesheet)
        font = self.message_label.font()
        font.setPointSize(20)
        font.setBold(True)
        self.message_label.setFont(font)
        self.message_label.setStyleSheet("color: white; background-color: #444; padding: 10px;")

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.message_label)
        
        self.registration_label = QLabel("Register New Student", self)
        self.registration_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.registration_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("Enter your name")

        self.register_btn = QPushButton("Register", self)
        self.register_btn.clicked.connect(self.on_register)

        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.addWidget(self.registration_label)
        self.right_layout.addWidget(self.name_input)
        self.right_layout.addWidget(self.register_btn)
        self.right_widget.hide()  # Hide until needed

        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.right_widget)
        self.setLayout(main_layout)

        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            QMessageBox.critical(self, "Webcam Error", "Could not access the webcam. Please check your device.")
            sys.exit(1)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.attendance_file = "attendance.csv"
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Time"])

        self.marked_today = set()

    def update_frame(self):
            
            ret, frame = self.capture.read()
            if not ret:
                return

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            display_name = "Unknown"

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                face_encoding = face_encoding.reshape(1, -1)
                face_scaled = scaler.transform(face_encoding)
                face_selected = selector.transform(face_scaled)

                if hasattr(classifier, "predict_proba"):
                    proba = classifier.predict_proba(face_selected)[0]
                    confidence = np.max(proba)
                    predicted_index = np.argmax(proba)
                    name = predicted_index if confidence > 0.7 else "Unknown"
                else:
                    name = classifier.predict(face_selected)[0]
                    confidence = None

                prediction_buffer.append(name)
                most_common = Counter(prediction_buffer).most_common(1)[0]
                stable_name, count = most_common

                if stable_name != "Unknown" and count >= 5:
                    display_name = label_map[stable_name]
                    if display_name not in self.marked_today:
                        self.mark_attendance(display_name)
                        self.marked_today.add(display_name)
                    self.message_label.setText(f"Welcome, {display_name}")
                    self.right_widget.hide()
                elif stable_name == "Unknown" and count >= 5:
                        self.message_label.setText("Face not recognized. Register?")
                        self.right_widget.show()


            # Draw box and label
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                label = f"{display_name}" if confidence else display_name
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Convert to Qt image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_frame.shape
            bytes_per_line = channel * width
            qt_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))


    def mark_attendance(self, name):
        with open(self.attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            
    def on_register(self):
        student_name = self.name_input.text().strip()
        if not student_name:
            QMessageBox.warning(self, "Error", "Please enter a name.")
            return

        save_path = capture_images(student_name)
        register_new_student(student_name, save_path)
        resplit_data()
        retrain_model()
        QMessageBox.information(self, "Registered", f"{student_name} has been registered.")

        self.name_input.clear()
        self.right_widget.hide()
        self.reload_data()
        print("[INFO] reload successful")

    def reload_data(self):
        global known_encodings, known_labels, label_map, reverse_map
        with open('processed_data.pkl', 'rb') as f:
            data = pickle.load(f)
            known_encodings = data['X']
            known_labels = data['y']
            label_map = data['label_map']
            reverse_map = {v: k for k, v in label_map.items()}


if __name__ == '__main__':
    print("[INFO] UI running....")
    app = QApplication(sys.argv)
    window = FaceAttendanceApp()
    window.show()
    sys.exit(app.exec())
