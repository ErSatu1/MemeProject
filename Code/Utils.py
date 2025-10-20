import cv2
import mediapipe as mp
import os

# === Setup directories ===
save_dir = "dataset"
os.makedirs(save_dir, exist_ok=True)
dataset_type = 'static'

# === Initialize MediaPipe modules ===
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# === Initialize webcam ===
cap = cv2.VideoCapture(0)
counter = 0  # image counter

print("Press 's' to save image, 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # --- FACE DETECTION ---
    face_results = face_detection.process(img_rgb)
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                                 int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(frame, 'Face', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- HAND DETECTION ---
    hand_results = hands.process(img_rgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )

            # Compute hand bounding box
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(frame, 'Hand', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # === Show frame ===
    cv2.imshow("Dataset Capture (Press 's' to Save, 'q' to Quit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Example: dataset_type = "face/train" or "hand/test"
        img_name = f"{save_dir}/{dataset_type}/capture_{counter:04d}.jpg"
        
        # Create all folders automatically
        os.makedirs(os.path.dirname(img_name), exist_ok=True)
        
        # Save image
        success = cv2.imwrite(img_name, frame)
        if success:
            print(f"✅ Saved: {img_name}")
            counter += 1
        else:
            print(f"❌ Failed to save: {img_name}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()