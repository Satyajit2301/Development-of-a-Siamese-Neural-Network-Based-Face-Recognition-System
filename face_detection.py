"""
face_detection.py
-----------------
Face detection and extraction with OpenCV's Haar Cascade classifier
(the Viola-Jones method).

Pipeline: frame -> grayscale -> detectMultiScale -> bounding boxes
          -> crop (with margin, 92:112 aspect) -> resize 112x92 -> normalise
"""
import os
import json
import cv2
import numpy as np

FACE_W, FACE_H = 92, 112          # model input size (width, height)
DEFAULT_MARGIN = 0.25             # extra context around the Haar box (fraction of box size)


def load_face_detector(cascade_path=None):
    """Load the pretrained Haar cascade XML file and return the classifier."""
    if cascade_path is None:
        cascade_path = os.path.join(cv2.data.haarcascades,
                                    "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise IOError(f"Could not load Haar cascade: {cascade_path}")
    return detector


def to_gray(frame):
    """Convert a BGR frame to grayscale (grayscale input is returned unchanged)."""
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def detect_faces(frame, detector, scale_factor=1.1, min_neighbors=5, min_size=(60, 60)):
    """
    Detect all faces in a frame.

    Returns
    -------
    list of (x, y, w, h) boxes sorted by area, largest face first.
    An empty list means that no face was found.
    """
    gray = to_gray(frame)
    boxes = detector.detectMultiScale(gray, scaleFactor=scale_factor,
                                      minNeighbors=min_neighbors, minSize=min_size)
    boxes = [tuple(int(v) for v in b) for b in boxes]
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "crop_calibration.json")
# fallback = old behaviour (margin 0.25, centred on the Haar box)
_DEFAULT_CALIBRATION = {"height_ratio": 1.5, "dx": 0.0, "dy": 0.0}
_CAL_CACHE = None


def load_calibration():
    """
    Geometry that maps a Haar box onto the framing of the training images
    (measured from the training identities, stored in crop_calibration.json).
      height_ratio : crop height / Haar box height
      dx, dy       : shift of the crop centre from the box centre (in box heights)
    """
    global _CAL_CACHE
    if _CAL_CACHE is None:
        try:
            with open(CALIBRATION_FILE) as f:
                _CAL_CACHE = json.load(f)
        except (OSError, ValueError):
            _CAL_CACHE = dict(_DEFAULT_CALIBRATION)
    return _CAL_CACHE


def crop_face(frame, box, margin=None):
    """
    Crop a face from a frame and resize it to 112x92 (grayscale, uint8).

    margin=None  -> use the calibrated geometry (matches the framing of the
                    AT&T training images; default).
    margin=float -> symmetric margin around the Haar box (used for comparison).
    """
    gray = to_gray(frame)
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    if margin is None:
        cal = load_calibration()
        new_h = h * cal["height_ratio"]
        cx += cal["dx"] * h
        cy += cal["dy"] * h
    else:
        new_h = h * (1 + 2 * margin)
    new_w = new_h * FACE_W / FACE_H
    x1, x2 = int(round(cx - new_w / 2)), int(round(cx + new_w / 2))
    y1, y2 = int(round(cy - new_h / 2)), int(round(cy + new_h / 2))
    pad = int(max(new_w, new_h)) + 2                     # keep crops near the border valid
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = padded[y1 + pad:y2 + pad, x1 + pad:x2 + pad]
    return cv2.resize(crop, (FACE_W, FACE_H))


def preprocess_face(face_gray):
    """Normalise to [0, 1] and add the channel axis -> float32 (112, 92, 1)."""
    face = face_gray.astype("float32") / 255.0
    return face.reshape(FACE_H, FACE_W, 1)


def get_face(frame, detector, margin=None, **detect_kwargs):
    """
    Full detection pipeline for one frame.

    Returns
    -------
    face    : float32 (112, 92, 1) of the LARGEST face, or None if no face
    box     : (x, y, w, h) of that face, or None
    n_faces : number of faces detected
    message : human-readable status for the interface
    """
    boxes = detect_faces(frame, detector, **detect_kwargs)
    if len(boxes) == 0:
        return None, None, 0, "No face detected"
    face = preprocess_face(crop_face(frame, boxes[0], margin))
    if len(boxes) == 1:
        msg = "Face detected"
    else:
        msg = f"{len(boxes)} faces detected - using the largest"
    return face, boxes[0], len(boxes), msg


def draw_detections(frame, boxes, thickness=2):
    """Return a BGR copy of the frame with boxes drawn (largest = green, others = orange)."""
    out = frame.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(boxes):
        color = (0, 255, 0) if i == 0 else (0, 165, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def run_webcam_detection(camera_index=0):
    """
    Live webcam demo (run on a local computer, not in Colab).
    Shows bounding boxes in real time. Press 'q' to quit.
    """
    detector = load_face_detector()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = detect_faces(frame, detector)
        shown = draw_detections(frame, boxes)
        text = "No face detected" if not boxes else f"{len(boxes)} face(s)"
        cv2.putText(shown, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Face detection (q to quit)", shown)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_detection()
