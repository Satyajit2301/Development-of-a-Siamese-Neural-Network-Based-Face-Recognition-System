"""
face_recognition_system.py
--------------------------
Real-time face verification / authentication.

Pipeline for each frame:
  detect faces (Viola-Jones) -> crop + preprocess -> 128-d embedding
  -> Euclidean distance to the registered users' embeddings
  -> user score = mean of the 3 smallest distances to that user's embeddings
  -> closest user is accepted only if score < threshold, otherwise "Unknown".
"""
import os
import json

import cv2
import numpy as np

import face_detection
from database import UserDatabase
from onboarding import load_embedding_model, embed_faces

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLD_FILE = os.path.join(BASE_DIR, "models", "threshold.json")
DEFAULT_THRESHOLD = 0.63
DEFAULT_METHOD = "knn3"      # one of: min, mean, knn3, centroid


def load_threshold(path=THRESHOLD_FILE, default=DEFAULT_THRESHOLD):
    """Read the validation-selected decision threshold (falls back to the default)."""
    try:
        with open(path) as f:
            return float(json.load(f)["threshold"])
    except (OSError, ValueError, KeyError):
        return default


def euclidean(a, b):
    """Euclidean distance between embeddings (same formula as the training code)."""
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    return np.sqrt(np.maximum(np.sum((a - b) ** 2, axis=-1), 1e-7))


def score_user(user_embeddings, query, method=DEFAULT_METHOD):
    """
    Dissimilarity between one query embedding and ALL stored embeddings of a user.
      min      : distance to the closest stored embedding
      mean     : mean distance to all stored embeddings
      knn3     : mean of the 3 smallest distances (default)
      centroid : distance to the average of the stored embeddings
    """
    user_embeddings = np.asarray(user_embeddings, dtype="float32")
    if method == "centroid":
        return float(euclidean(query, user_embeddings.mean(axis=0)))
    d = np.sort(euclidean(query[None, :], user_embeddings))
    if method == "min":
        return float(d[0])
    if method == "mean":
        return float(d.mean())
    if method == "knn3":
        return float(d[:3].mean())
    raise ValueError(f"Unknown scoring method: {method}")


class FaceRecognizer:
    """Compare faces in webcam frames with the registered users."""

    def __init__(self, embedder, detector, db, threshold=None, method=DEFAULT_METHOD):
        self.embedder = embedder
        self.detector = detector
        self.db = db
        self.threshold = load_threshold() if threshold is None else float(threshold)
        self.method = method
        self.registry = {}
        self.refresh()

    def refresh(self):
        """Reload the registered users' embeddings from the database."""
        reg = self.db.get_all_embeddings()
        self.registry = {u: v for u, v in reg.items() if len(v["embeddings"]) > 0}

    def identify(self, embedding):
        """
        Compare one embedding with every registered user.

        Returns a dict: recognized (bool), user_id / name (None when unknown),
        closest_user_id, distance (score of the closest user), scores {user_id: score}.
        """
        if not self.registry:
            return {"recognized": False, "user_id": None, "name": None,
                    "closest_user_id": None, "distance": None, "scores": {}}
        scores = {u: score_user(v["embeddings"], embedding, self.method)
                  for u, v in self.registry.items()}
        closest = min(scores, key=scores.get)
        dist = scores[closest]
        ok = dist < self.threshold
        return {"recognized": bool(ok),
                "user_id": closest if ok else None,
                "name": self.registry[closest]["name"] if ok else None,
                "closest_user_id": closest, "distance": float(dist), "scores": scores}

    def recognize_frame(self, frame):
        """
        Run the full pipeline on one frame.

        Handles both full webcam/scene frames and pre-cropped face images.

        Returns a dict:
          status  : "recognized" | "unknown" | "no_face" | "no_users"
          message : text for the interface
          faces   : one result dict (with its box) per detected face, largest first
          primary : result of the largest face (None if no face)
        The authentication decision uses the largest face in the frame.
        """
        if not self.registry:
            return {"status": "no_users", "message": "No registered users - please register first", "faces": [], "primary": None}

        h, w = frame.shape[:2]
        boxes = face_detection.detect_faces(frame, self.detector)

        # Check if this image is already a pre-cropped face (e.g., uploaded avatar, saved crop)
        is_pre_cropped = (w <= 220 and h <= 250) or (len(boxes) == 1 and (boxes[0][2] * boxes[0][3]) / (w * h) > 0.50 and w <= 320)

        if is_pre_cropped or len(boxes) == 0:
            # Evaluate direct crop (resize directly to 112x92 without destructive double-cropping)
            gray = face_detection.to_gray(frame)
            face_direct = cv2.resize(gray, (face_detection.FACE_W, face_detection.FACE_H))
            prepped = face_detection.preprocess_face(face_direct)
            emb = embed_faces(self.embedder, [prepped])[0]
            r = self.identify(emb)

            # If no Haar box was found and direct distance is above threshold, it's not a face
            if len(boxes) == 0 and not r["recognized"] and (r["distance"] is None or r["distance"] >= self.threshold):
                return {"status": "no_face", "message": "No face detected", "faces": [], "primary": None}

            r["box"] = (0, 0, w, h)
            primary = r
            results = [r]
            if primary["recognized"]:
                status = "recognized"
                msg = f"Recognized: {primary['name']} (distance {primary['distance']:.2f})"
            else:
                status = "unknown"
                msg = f"Unknown User (closest distance {primary['distance']:.2f}, threshold {self.threshold:.2f})"
            return {"status": status, "message": msg, "faces": results, "primary": primary}

        crops = [face_detection.crop_face(frame, b) for b in boxes]
        faces = np.array([face_detection.preprocess_face(c) for c in crops])
        embeddings = embed_faces(self.embedder, faces)

        results = []
        for box, emb in zip(boxes, embeddings):
            r = self.identify(emb)
            r["box"] = box
            results.append(r)
        primary = results[0]

        if primary["recognized"]:
            status = "recognized"
            msg = f"Recognized: {primary['name']} (distance {primary['distance']:.2f})"
        else:
            status = "unknown"
            msg = f"Unknown User (closest distance {primary['distance']:.2f}, threshold {self.threshold:.2f})"
        if len(boxes) > 1:
            msg += f" | {len(boxes)} faces in frame - decision uses the largest"
        return {"status": status, "message": msg, "faces": results, "primary": primary}


def annotate_frame(frame, result):
    """Draw boxes, names / distances and a status banner on a copy of the frame."""
    out = frame.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for r in result["faces"]:
        x, y, w, h = r["box"]
        color = (0, 200, 0) if r["recognized"] else (0, 0, 255)
        label = r["name"] if r["recognized"] else "Unknown"
        if r["distance"] is not None:
            label += f" ({r['distance']:.2f})"
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, label, (x, max(y - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    banner = {"recognized": (0, 200, 0), "unknown": (0, 0, 255),
              "no_face": (0, 140, 255), "no_users": (0, 140, 255)}[result["status"]]
    cv2.putText(out, result["message"][:70], (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, banner, 2)
    return out


def run_webcam_recognition(camera_index=0):
    """Live recognition on a local computer. Press 'q' to quit."""
    embedder = load_embedding_model()
    detector = face_detection.load_face_detector()
    rec = FaceRecognizer(embedder, detector, UserDatabase())
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Face recognition (q to quit)", annotate_frame(frame, rec.recognize_frame(frame)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_recognition()
