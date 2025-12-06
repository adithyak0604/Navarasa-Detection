import cv2
import numpy as np
import tensorflow as tf
import time
import sys

# -------------------------------
# Load Your Keras .h5 Model
# -------------------------------
model = tf.keras.models.load_model("navarasa_emotion_model_split1.h5")

# -------------------------------
# Class Labels
# -------------------------------
class_labels = [
    'ADBHUTA', 'BHAYANAKA', 'BIBHATSYA', 'HASYA', 'KARUNA',
    'RAUDRA', 'SHANTA', 'SHRINGARA', 'VEERA'
]

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_face(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    tensor = normalized.reshape(1, 48, 48, 1)
    return tensor

# -------------------------------
# Haarcascade Face Detector
# -------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detection lock config
LOCK_DELAY_SECONDS = 3
locked = False
lock_time = None
locked_face = None
locked_emotion = None

# -------------------------------
# Video capture helper
# -------------------------------
'''cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()'''


def get_video_capture(source):
    """
    source:
      - int (0,1,...) to use local webcam device index
      - string URL like "http://192.168.x.x:8080/video" or "rtsp://..."
    """
    try:
        # if source looks like a number, convert to int
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)
        # some IP streams require setting a backend; uncomment if needed:
        # cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        return cap
    except Exception as e:
        print("get_video_capture error:", e)
        return None

# -------------------------------
# MAIN
# -------------------------------
# Usage: python 1.py [video_source]
# Examples:
#   python 1.py 0                    -> use local webcam index 0
#   python 1.py http://192.168.0.101:8080/video  -> use IP Webcam stream from phone
if len(sys.argv) > 1:
    src = sys.argv[1]
else:
    # default to local webcam 0 if no argument provided
    src = "0"

cap = get_video_capture(src)
if cap is None or not cap.isOpened():
    print(f"Error: Could not open video source '{src}'.")
    print("If using a phone stream, check URL and ensure phone + PC are on same Wi-Fi.")
    sys.exit(1)

print("Press 'q' to exit | Press 'r' to unlock")

reconnect_attempts = 0
MAX_RECONNECT = 5

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        # try reconnecting for IP streams
        reconnect_attempts += 1
        print(f"Warning: failed to read frame (attempt {reconnect_attempts}/{MAX_RECONNECT}). Reopening source...")
        cap.release()
        time.sleep(0.8)
        cap = get_video_capture(src)
        if cap is None:
            print("Could not reopen capture.")
            break
        if reconnect_attempts > MAX_RECONNECT:
            print("Exceeded max reconnect attempts. Exiting.")
            break
        continue
    reconnect_attempts = 0  # reset on success

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If locked, just display the locked emotion + box
    if locked and locked_face is not None:
        (x, y, w, h) = locked_face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
        cv2.putText(frame, f"{locked_emotion} (LOCKED)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        cv2.imshow("Navarasa Emotion Detection", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('r'):
            locked = False
            locked_face = None
            locked_emotion = None
            lock_time = None
        continue

    # Detect faces (only if not locked)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        lock_time = None
    else:
        # Take the first detected face
        (x, y, w, h) = faces[0]

        # Start lock timer if not already started
        if lock_time is None:
            lock_time = time.time()

        # Draw countdown timer
        elapsed = time.time() - lock_time
        remaining = max(0, LOCK_DELAY_SECONDS - elapsed)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Locking in: {remaining:.1f}s", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # If timer completes → LOCK the face
        if elapsed >= LOCK_DELAY_SECONDS:
            locked = True
            locked_face = (x, y, w, h)

            # Predict emotion ONCE
            face_roi = frame[y:y+h, x:x+w]
            input_tensor = preprocess_face(face_roi)
            preds = model.predict(input_tensor)
            idx = np.argmax(preds)
            locked_emotion = class_labels[idx]

            print(f"[LOCKED] Emotion = {locked_emotion}")
            continue  # next loop, locked mode

    # Display output in unlock mode
    cv2.imshow("Navarasa Emotion Detection", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('r'):
        locked = False
        locked_face = None
        locked_emotion = None
        lock_time = None

cap.release()
cv2.destroyAllWindows()
