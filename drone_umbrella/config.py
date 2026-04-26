# ============================================================
# config.py — All the knobs and dials for the whole system
#
# If something feels off (drone overshoots, moves wrong way,
# camera is wrong index, etc.) — this is the only file you
# need to touch. Everything else reads from here.
# ============================================================

import cv2

# ── Camera & Window ─────────────────────────────────────────
WEBCAM_INDEX  = 0      # Try 1 if your default webcam doesn't open
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
SIM_SIZE      = 500    # Pixel size of the top-down simulation window

# ── Drone Movement Speeds (0–100, Tello units) ───────────────
SPEED_LR  = 35   # Left/right strafe speed
SPEED_FB  = 25   # Forward/backward speed
SPEED_UD  = 0    # Up/down (we keep altitude fixed)
SPEED_YAW = 0    # Rotation (we don't rotate, just strafe)

# How often we fire RC commands to the Tello (20 Hz is its sweet spot)
RC_INTERVAL = 0.05

# ── Face Tracking Tuning ─────────────────────────────────────
# TARGET_AREA is the face pixel area we consider "ideal distance".
# Measured from a real screenshot — tweak if your drone hovers too
# close or too far.
TARGET_AREA  = 17000

# The drone won't move forward/back unless the depth error exceeds this.
# Large deadzone = less jitter, but slower depth correction.
FB_DEADZONE  = 0.20

# Even if error is tiny, the Tello needs at least this speed to actually move
FB_MIN_SPEED = 15

# Same idea for left/right — smaller deadzone so horizontal tracking is snappy
LR_DEADZONE  = 0.05

# Laptop camera and Tello both face you, so drone-right = your screen-left.
# Set to True to compensate for that mirror flip.
# If the drone still drifts the wrong way, flip this to False.
LR_INVERT = True

# ── Smoothing Factors (0 = frozen, 1 = instant snap) ─────────
PERSON_SMOOTH = 0.4    # How quickly the "tracked person" position updates
DRONE_SMOOTH  = 0.15   # How smoothly the sim drone icon follows in the display

# ── Face Detector ────────────────────────────────────────────
# OpenCV's built-in Haar cascade — no extra download needed
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── UI Colors (BGR format, because OpenCV is old-school) ─────
CLR_DRONE    = (50, 50, 220)     # Blue-ish drone icon
CLR_PERSON   = (220, 120, 50)    # Orange person icon
CLR_LINE     = (160, 160, 160)   # Line connecting drone to person
CLR_BOX      = (50, 220, 100)    # Green face bounding box
CLR_TEXT     = (240, 240, 240)   # Default white text
CLR_PANEL    = (30, 30, 30)      # Dark panel background
CLR_GRID     = (60, 60, 60)      # Simulation grid lines
CLR_TRACKING = (50, 200, 100)    # Green = actively tracking
CLR_LOST     = (50, 80, 220)     # Blue = face lost
CLR_WARN     = (30, 160, 220)    # Yellow-ish warning / airborne indicator
