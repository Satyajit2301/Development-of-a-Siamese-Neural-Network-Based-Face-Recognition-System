"""
onboarding.py
-------------
User enrolment: capture several face photos, detect and crop the face in
each one, generate a 128-d embedding with the trained shared CNN and store
the user's record in the database.

Rules used at enrolment
  * a frame is used only if it contains EXACTLY ONE face
    (frames with no face or several faces are skipped, so a bystander cannot
    be enrolled by accident)
  * at least MIN_FACES usable faces are required
"""
import os
import sys
import time

import cv2
import keras
import numpy as np

import face_detection
from database import UserDatabase, is_valid_user_id

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EMBEDDER_PATH = os.path.join(BASE_DIR, "models", "shared_cnn.h5")
NUM_CAPTURES = 10      # photos captured during webcam enrolment
MIN_FACES = 5          # minimum usable faces needed to register a user


def load_embedding_model(path=DEFAULT_EMBEDDER_PATH):
    """Load the trained shared CNN (image -> 128-d embedding)."""
    return keras.models.load_model(path, compile=False)


def embed_faces(embedder, faces):
    """Embeddings for preprocessed faces: float32 (n, 112, 92, 1) -> (n, 128)."""
    faces = np.asarray(faces, dtype="float32")
    if faces.ndim == 3:
        faces = faces[None]
    return embedder.predict(faces, verbose=0)


def enrol_from_frames(user_id, name, frames, embedder, detector, db, min_faces=MIN_FACES):
    """
    Register a user from a list of BGR (or grayscale) frames.

    Returns a dictionary with: success (bool), message (str), frames_received,
    faces_used, skipped_no_face, skipped_multiple_faces.
    """
    result = {"success": False, "message": "", "frames_received": len(frames),
              "faces_used": 0, "skipped_no_face": 0, "skipped_multiple_faces": 0}

    user_id = (user_id or "").strip()
    name = (name or "").strip()

    if not user_id:
        result["message"] = "Please enter a User ID"
        return result
    if not is_valid_user_id(user_id):
        result["message"] = "Invalid user ID: use 1-32 letters, digits, '_' or '-'"
        return result
    if not name:
        result["message"] = "Please enter a full name"
        return result
    if db.user_exists(user_id):
        result["message"] = f"User ID '{user_id}' is already registered"
        return result

    faces, crops = [], []
    for frame in frames:
        boxes = face_detection.detect_faces(frame, detector)
        if len(boxes) == 0:
            result["skipped_no_face"] += 1
            continue
        if len(boxes) > 1:
            result["skipped_multiple_faces"] += 1
            continue
        crop = face_detection.crop_face(frame, boxes[0])
        crops.append(crop)
        faces.append(face_detection.preprocess_face(crop))

    result["faces_used"] = len(faces)
    if len(faces) < min_faces:
        result["message"] = (f"Only {len(faces)} usable face(s) found "
                             f"(need at least {min_faces}). Face the camera and try again.")
        return result

    embeddings = embed_faces(embedder, faces)
    db.add_user(user_id, name, embeddings, crops)
    result["success"] = True
    result["message"] = f"User '{name.strip()}' registered with {len(faces)} face embeddings"
    return result


def capture_frames_webcam(n_frames=NUM_CAPTURES, camera_index=0, interval=0.6):
    """
    Capture enrolment frames from a local webcam (run on your own computer).
    A frame is captured automatically every `interval` seconds while exactly
    one face is visible. Turn your head slightly between captures. Press 'q' to stop.
    """
    detector = face_detection.load_face_detector()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam")
    frames, last = [], 0.0
    while len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = face_detection.detect_faces(frame, detector)
        shown = face_detection.draw_detections(frame, boxes)
        if len(boxes) == 1:
            text, color = f"Capturing {len(frames)}/{n_frames}", (0, 255, 0)
            if time.time() - last >= interval:
                frames.append(frame.copy())
                last = time.time()
        elif len(boxes) == 0:
            text, color = "No face detected", (0, 0, 255)
        else:
            text, color = "Multiple faces - only one person please", (0, 165, 255)
        cv2.putText(shown, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Enrolment (q to stop)", shown)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frames


def register_user_webcam(user_id, name, n_frames=NUM_CAPTURES, camera_index=0):
    """Full enrolment from the webcam: capture -> detect -> embed -> store."""
    embedder = load_embedding_model()
    detector = face_detection.load_face_detector()
    db = UserDatabase()
    frames = capture_frames_webcam(n_frames, camera_index)
    result = enrol_from_frames(user_id, name, frames, embedder, detector, db)
    print(result["message"])
    return result


if __name__ == "__main__":
    # python onboarding.py <user_id> "<Full Name>"
    if len(sys.argv) < 3:
        print('Usage: python onboarding.py <user_id> "<Full Name>"')
    else:
        register_user_webcam(sys.argv[1], sys.argv[2])
