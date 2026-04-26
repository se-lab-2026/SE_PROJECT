# ============================================================
# detection.py — Face detection using OpenCV Haar cascades
#
# We grab the largest detected face each frame (so if multiple
# faces appear, the drone tracks the most prominent one).
# Returns normalized center coordinates + pixel area so the
# control logic can compute how far off-target we are.
# ============================================================

import cv2
from config import face_cascade


def detect_face(frame):
    """
    Scans a single BGR frame for faces and returns info about
    the largest one found (most likely the primary subject).

    Returns:
        cx   (int|None) — pixel X of face center, None if no face
        cy   (int|None) — pixel Y of face center
        area (int)      — bounding box area in pixels (0 if none)
        bbox (tuple)    — (x, y, w, h) of the face rect, None if no face
    """
    # Haar cascades work on grayscale — color info just slows it down
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,    # How much the image is scaled down each pass
        minNeighbors=5,     # Higher = fewer false positives but may miss faces
        minSize=(60, 60)    # Ignore tiny blobs that probably aren't faces
    )

    if len(faces) > 0:
        # If multiple faces detected, track the biggest one (closest person)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cx   = x + w // 2
        cy   = y + h // 2
        area = w * h
        return cx, cy, area, (x, y, w, h)

    # No face found this frame — caller handles the "lost" case
    return None, None, 0, None
