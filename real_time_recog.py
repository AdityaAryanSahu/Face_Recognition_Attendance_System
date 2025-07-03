import cv2
import face_recognition
import pickle
import numpy as np
from collections import deque, Counter

# this can be used to check if your model is working and predicting 
# accurately using live webcam



# Load saved models and preprocessing tools
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)
label_map = data['label_map'] 

with open("voting_classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("selector.pkl", "rb") as f:
    selector = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Create prediction buffer for smoothing
prediction_buffer = deque(maxlen=5)

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting video stream...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and compute encodings
    face_locations = face_recognition.face_locations(rgb_small_frame,model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_encoding = face_encoding.reshape(1, -1)
        face_scaled = scaler.transform(face_encoding)
        face_selected = selector.transform(face_scaled)

        # Predict
        if hasattr(classifier, "predict_proba"):
            proba = classifier.predict_proba(face_selected)[0]
            confidence = np.max(proba)
            predicted_index = np.argmax(proba)
           # print(f"[DEBUG] Predicted: {label_map[predicted_index]}, Confidence: {confidence:.3f}")
            if confidence > 0.70:
                name = predicted_index
            else:
                name = "Unknown"
        else:
            name = classifier.predict(face_selected)[0]
            confidence = None

        # Buffer-based smoothing
        prediction_buffer.append(name)
        most_common = Counter(prediction_buffer).most_common(1)[0]
        stable_name, count = most_common

        # Display if prediction is stable
        if stable_name != "Unknown" and count >= 3:
            display_name = label_map[stable_name]
        else:
            display_name = "Unknown"

        # Scale coordinates back to original size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{display_name}{confidence}" if confidence else display_name
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Real-Time Face Recognition - Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Video stream stopped.")
