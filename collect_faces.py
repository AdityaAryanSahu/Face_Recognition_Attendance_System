import cv2
import os
import face_recognition

# this module is to make it easier to save the images of newly registered members

def capture_images(name):
    # update this path with wherever you wanna save the new data
    save_dir = os.path.join(r"C:\Users\Lenovo\Downloads\Image_dataset", name) 

    # Create the folder if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print("[INFO] Starting webcam...")

    img_count = 0
    max_images = 20 # change the limit to whatever number you want 

    while True:
        ret, frame = cap.read()
        if not ret:
            break
    

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        face_locations = face_recognition.face_locations(small_frame)

        for top, right, bottom, left in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Extract and save face region
            face_img = frame[top:bottom, left:right]
            if face_img.size > 0:
                img_path = os.path.join(save_dir, f"{img_count+1}.jpg")
                cv2.imwrite(img_path, face_img)
                img_count += 1

            cv2.putText(frame, f"Images: {img_count}/{max_images}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Collector - Press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or img_count >= max_images:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Collected {img_count} face images for '{name}' in folder: {save_dir}")
    return save_dir
    
    