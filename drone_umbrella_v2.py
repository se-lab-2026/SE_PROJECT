"""
Autonomous Drone Umbrella System
=================================
Controls:
  SPACE  - Takeoff / Land toggle
  S      - Start tracking
  P      - Pause tracking
  R      - Reset simulation
  Q      - Emergency land + quit
  D      - Toggle drone mode ON/OFF (OFF = simulation only)

Requirements:
  pip install opencv-python djitellopy numpy

How to connect drone:
  1. Power on your Tello drone
  2. Connect your laptop Wi-Fi to the Tello hotspot (TELLO_XXXXXXX)
  3. Run this script with drone mode ON (press D to toggle)
  4. Press SPACE to takeoff, S to start tracking
"""

import cv2
import numpy as np
import time
import threading

# ── Try importing Tello. If not installed, simulation-only mode ──
try:
    from djitellopy import Tello
    TELLO_AVAILABLE = True
except ImportError:
    TELLO_AVAILABLE = False
    print("[WARNING] djitellopy not found. Running in simulation-only mode.")
    print("[INFO]    Install with: pip install djitellopy")


# ════════════════════════════════════════════════
#  SETTINGS  — tweak these for your environment
# ════════════════════════════════════════════════
WEBCAM_INDEX      = 0       # Try 1 if webcam doesn't open
FRAME_WIDTH       = 640
FRAME_HEIGHT      = 480
SIM_SIZE          = 500     # Simulation window size (square)

# Color to track — default is GREEN (good for a green shirt / object)
# Use the HSV Tuner at the bottom to find your color
COLOR_LOWER = np.array([0, 45, 0])  # HSV lower bound
COLOR_UPPER = np.array([60, 255, 255]) # HSV upper bound
MIN_AREA    = 1500          # Ignore tiny blobs smaller than this

# Drone speed settings (0-100)
SPEED_LR    = 25            # Left/Right speed
SPEED_FB    = 0             # Forward/Back (keep 0 for umbrella hover)
SPEED_UD    = 0             # Up/Down (keep 0)
SPEED_YAW   = 0             # Rotation (keep 0)

# How sensitive the drone reacts to position error (lower = smoother)
KP = 0.25                   # Proportional gain


# ════════════════════════════════════════════════
#  COLORS for OpenCV drawing (BGR format)
# ════════════════════════════════════════════════
CLR_DRONE     = (50,  50, 220)   # Red-ish  — drone dot
CLR_PERSON    = (220, 120, 50)   # Blue-ish — person dot
CLR_LINE      = (160, 160, 160)  # Gray     — connection line
CLR_BOX       = (50,  220, 100)  # Green    — detection box
CLR_TEXT      = (240, 240, 240)  # White    — text
CLR_PANEL     = (30,   30,  30)  # Dark     — panel background
CLR_GRID      = (60,   60,  60)  # Gray     — grid lines
CLR_TRACKING  = (50,  200, 100)  # Green    — tracking status
CLR_LOST      = (50,  80,  220)  # Red      — lost status
CLR_WARN      = (30, 160, 220)   # Amber    — warning


# ════════════════════════════════════════════════
#  STATE
# ════════════════════════════════════════════════
class AppState:
    def __init__(self):
        self.tracking     = False
        self.airborne     = False
        self.drone_mode   = False      # False = simulation only
        self.running      = True

        # Positions (normalized 0.0–1.0 of frame)
        self.person_x     = 0.5
        self.person_y     = 0.5
        self.drone_x      = 0.5       # Drone sim position
        self.drone_y      = 0.2       # Drone hovers above person

        # Telemetry
        self.battery      = 0
        self.detected     = False
        self.fps          = 0
        self.error_x      = 0
        self.status_msg   = "Ready"

        self._lock        = threading.Lock()

    def set(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def get(self, key):
        with self._lock:
            return getattr(self, key)


# ════════════════════════════════════════════════
#  DRONE CONTROLLER
# ════════════════════════════════════════════════
class DroneController:
    def __init__(self, state: AppState):
        self.state  = state
        self.tello  = None
        self.active = False

    def connect(self):
        if not TELLO_AVAILABLE:
            print("[DRONE] djitellopy not installed — simulation only")
            return False
        try:
            print("[DRONE] Connecting to Tello...")
            self.tello = Tello()
            self.tello.connect()
            bat = self.tello.get_battery()
            self.state.set(battery=bat)
            self.active = True
            print(f"[DRONE] Connected! Battery: {bat}%")
            return True
        except Exception as e:
            print(f"[DRONE] Connection failed: {e}")
            print("[DRONE] Falling back to simulation mode")
            self.active = False
            return False

    def takeoff(self):
        if self.active and self.tello:
            try:
                print("[DRONE] Taking off...")
                self.tello.takeoff()
                self.state.set(airborne=True, status_msg="Airborne")
                # Update battery after takeoff
                time.sleep(0.5)
                self.state.set(battery=self.tello.get_battery())
            except Exception as e:
                print(f"[DRONE] Takeoff error: {e}")
        else:
            # Simulation takeoff
            self.state.set(airborne=True, status_msg="Airborne (sim)")

    def land(self):
        if self.active and self.tello:
            try:
                print("[DRONE] Landing...")
                self.tello.land()
                self.state.set(airborne=False, status_msg="Landed")
            except Exception as e:
                print(f"[DRONE] Land error: {e}")
        else:
            self.state.set(airborne=False, status_msg="Landed (sim)")

    def emergency_stop(self):
        if self.active and self.tello:
            try:
                self.tello.land()
            except:
                pass
        self.state.set(airborne=False, running=False)

    def send_tracking_command(self, error_x: float):
        """
        error_x: how far the person is from center (-1.0 left … +1.0 right)
        Drone moves left/right to keep person centered.
        """
        lr = int(np.clip(error_x * SPEED_LR / 0.5, -SPEED_LR, SPEED_LR))
        if self.active and self.tello and self.state.airborne:
            try:
                self.tello.send_rc_control(lr, SPEED_FB, SPEED_UD, SPEED_YAW)
            except Exception as e:
                print(f"[DRONE] RC error: {e}")

    def disconnect(self):
        if self.active and self.tello:
            try:
                if self.state.airborne:
                    self.tello.land()
                self.tello.end()
            except:
                pass


# ════════════════════════════════════════════════
#  COLOR TRACKER
# ════════════════════════════════════════════════
def detect_person(frame):
    """
    Detects colored object in frame.
    Returns: (cx, cy, area, bbox) or (None, None, 0, None)
    """
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask    = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)

    # Clean up mask
    kernel  = np.ones((5, 5), np.uint8)
    mask    = cv2.erode(mask,  kernel, iterations=2)
    mask    = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        if area >= MIN_AREA:
            x, y, w, h = cv2.boundingRect(largest)
            cx = x + w // 2
            cy = y + h // 2
            return cx, cy, area, (x, y, w, h)

    return None, None, 0, None


# ════════════════════════════════════════════════
#  DRAWING HELPERS
# ════════════════════════════════════════════════
def draw_panel(img, x, y, w, h, alpha=0.6):
    """Draw a semi-transparent dark panel."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), CLR_PANEL, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    cv2.rectangle(img, (x, y), (x+w, y+h), (80, 80, 80), 1)

def put_text(img, text, pos, color=CLR_TEXT, scale=0.55, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_status_overlay(frame, state: AppState):
    """Draw the HUD on the webcam feed."""
    h, w = frame.shape[:2]

    # Top panel
    draw_panel(frame, 0, 0, w, 56)

    tracking_color = CLR_TRACKING if state.detected else CLR_LOST
    tracking_label = "TRACKING" if state.detected else "LOST"

    put_text(frame, "Autonomous Drone Umbrella",       (10, 18),  CLR_TEXT,     0.55, 1)
    put_text(frame, f"FPS: {state.fps}",               (10, 38),  CLR_TEXT,     0.45)
    put_text(frame, f"State: {tracking_label}",        (180, 18), tracking_color, 0.55, 1)
    put_text(frame, f"Mode: {'DRONE' if state.drone_mode else 'SIM'}",
                                                       (180, 38), CLR_WARN if state.drone_mode else CLR_TEXT, 0.45)
    if state.airborne:
        put_text(frame, "AIRBORNE", (w-100, 18), CLR_WARN, 0.55, 2)
    put_text(frame, f"Battery: {state.battery}%",      (w-120, 38), CLR_TEXT,   0.45)

    # Bottom controls panel
    draw_panel(frame, 0, h-36, w, 36)
    put_text(frame, "SPACE:Takeoff/Land  S:Track  P:Pause  R:Reset  D:Toggle Drone  Q:Quit",
             (8, h-14), CLR_TEXT, 0.38)

    # Person position info
    if state.detected:
        put_text(frame, f"Person: ({int(state.person_x*w)}, {int(state.person_y*h)})",
                 (10, h-50), CLR_TRACKING, 0.45)
        put_text(frame, f"Error X: {state.error_x:+.2f}",
                 (200, h-50), CLR_WARN, 0.45)


def draw_simulation(state: AppState):
    """Draw the top-down 2D simulation window."""
    sim = np.ones((SIM_SIZE, SIM_SIZE, 3), dtype=np.uint8) * 245
    sim[:] = (245, 245, 242)  # Off-white background

    # Grid
    step = SIM_SIZE // 10
    for i in range(0, SIM_SIZE, step):
        cv2.line(sim, (i, 0), (i, SIM_SIZE), CLR_GRID, 1)
        cv2.line(sim, (0, i), (SIM_SIZE, i), CLR_GRID, 1)

    # Center crosshair
    cx = SIM_SIZE // 2
    cv2.line(sim, (cx, 0), (cx, SIM_SIZE), (180, 180, 180), 1)
    cv2.line(sim, (0, cx), (SIM_SIZE, cx), (180, 180, 180), 1)

    # Convert normalized positions to pixel positions
    px = int(state.person_x * SIM_SIZE)
    py = int(state.person_y * SIM_SIZE)
    dx = int(state.drone_x  * SIM_SIZE)
    dy = int(state.drone_y  * SIM_SIZE)

    # Connecting line (drone → person)
    cv2.line(sim, (dx, dy), (px, py), CLR_LINE, 2)

    # Shadow under person
    cv2.ellipse(sim, (px, py+22), (18, 6), 0, 0, 360, (200, 200, 200), -1)

    # Person dot
    cv2.circle(sim, (px, py), 18, CLR_PERSON, -1)
    cv2.circle(sim, (px, py), 18, (80, 60, 20), 2)
    put_text(sim, "P", (px-5, py+5), (255, 255, 255), 0.55, 2)

    # Drone dot (with propeller visual)
    for angle in [45, 135, 225, 315]:
        rad = np.radians(angle)
        ex  = int(dx + 16 * np.cos(rad))
        ey  = int(dy + 16 * np.sin(rad))
        cv2.line(sim, (dx, dy), (ex, ey), (100, 100, 100), 2)
        cv2.circle(sim, (ex, ey), 5, (140, 140, 140), -1)

    cv2.circle(sim, (dx, dy), 10, CLR_DRONE, -1)
    cv2.circle(sim, (dx, dy), 10, (10, 10, 120), 2)
    put_text(sim, "D", (dx-5, dy+5), (255, 255, 255), 0.45, 2)

    # Status panel (bottom)
    draw_panel(sim, 0, SIM_SIZE-90, SIM_SIZE, 90, alpha=0.75)

    tracking_col = CLR_TRACKING if state.detected else CLR_LOST
    put_text(sim, f"Tracking:  {'ACTIVE' if state.tracking else 'PAUSED'}",
             (10, SIM_SIZE-72), CLR_TEXT, 0.45)
    put_text(sim, f"Detection: {'YES' if state.detected else 'NO'}",
             (10, SIM_SIZE-54), tracking_col, 0.45)
    put_text(sim, f"Person:    ({state.person_x:.2f}, {state.person_y:.2f})",
             (10, SIM_SIZE-36), CLR_TEXT, 0.45)
    put_text(sim, f"Drone:     ({state.drone_x:.2f}, {state.drone_y:.2f})",
             (10, SIM_SIZE-18), CLR_TEXT, 0.45)
    put_text(sim, f"Status:    {state.status_msg}",
             (SIM_SIZE//2+10, SIM_SIZE-54), CLR_WARN, 0.45)
    put_text(sim, f"Mode:      {'DRONE' if state.drone_mode else 'SIMULATION'}",
             (SIM_SIZE//2+10, SIM_SIZE-36), CLR_TEXT, 0.45)
    put_text(sim, f"Battery:   {state.battery}%",
             (SIM_SIZE//2+10, SIM_SIZE-18), CLR_TEXT, 0.45)

    # Title
    draw_panel(sim, 0, 0, SIM_SIZE, 30, alpha=0.7)
    put_text(sim, "Drone Umbrella  —  Top-down simulation", (10, 20), CLR_TEXT, 0.48)

    return sim


# ════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("  Autonomous Drone Umbrella System")
    print("=" * 55)
    print(f"  Tracking color HSV: {COLOR_LOWER} → {COLOR_UPPER}")
    print(f"  Press D to enable drone mode after connecting Wi-Fi")
    print("=" * 55)

    state      = AppState()
    drone_ctrl = DroneController(state)

    # Open webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam index {WEBCAM_INDEX}. Try changing WEBCAM_INDEX to 1.")
        return

    print("[OK] Webcam opened.")

    cv2.namedWindow("Webcam Feed",      cv2.WINDOW_NORMAL)
    cv2.namedWindow("Simulation View",  cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam Feed",     FRAME_WIDTH, FRAME_HEIGHT)
    cv2.resizeWindow("Simulation View", SIM_SIZE,    SIM_SIZE)
    cv2.moveWindow("Webcam Feed",       0,           0)
    cv2.moveWindow("Simulation View",   FRAME_WIDTH + 10, 0)

    fps_timer  = time.time()
    fps_count  = 0
    smooth_x   = 0.5
    smooth_y   = 0.5

    print("[OK] Windows created. Controls active.")

    while state.running:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read webcam frame.")
            break

        frame = cv2.flip(frame, 1)   # Mirror so left/right feel natural
        fh, fw = frame.shape[:2]

        # ── Detection ───────────────────────────────────────
        cx, cy, area, bbox = detect_person(frame)
        detected = cx is not None

        if detected:
            # Normalize to 0.0–1.0
            nx = cx / fw
            ny = cy / fh

            # Smooth the movement
            smooth_x = smooth_x * 0.6 + nx * 0.4
            smooth_y = smooth_y * 0.6 + ny * 0.4

            error_x = (smooth_x - 0.5) * 2.0   # -1.0 to +1.0

            state.set(
                person_x = smooth_x,
                person_y = smooth_y,
                detected = True,
                error_x  = error_x,
                status_msg = "Tracking"
            )

            # Draw detection box on webcam feed
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), CLR_BOX, 2)
                cv2.circle(frame, (cx, cy), 6, CLR_BOX, -1)
                put_text(frame, f"Area: {int(area)}", (x, y-8), CLR_BOX, 0.4)

            # ── Send drone command ───────────────────────────
            if state.tracking and state.airborne:
                drone_ctrl.send_tracking_command(error_x)

        else:
            state.set(detected=False, status_msg="Searching..." if state.tracking else "Paused")
            if state.tracking and state.airborne:
                # Stop moving when lost
                drone_ctrl.send_tracking_command(0)

        # ── Simulate drone position ──────────────────────────
        # Drone smoothly follows person horizontally, stays above
        state.drone_x = state.drone_x * 0.85 + state.person_x * 0.15
        state.drone_y = 0.18           # Always hover at top

        # ── FPS counter ─────────────────────────────────────
        fps_count += 1
        if time.time() - fps_timer >= 1.0:
            state.set(fps=fps_count)
            fps_count = 0
            fps_timer = time.time()

        # ── Draw overlays ────────────────────────────────────
        draw_status_overlay(frame, state)
        sim = draw_simulation(state)

        cv2.imshow("Webcam Feed",     frame)
        cv2.imshow("Simulation View", sim)

        # ── Key handling ─────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:       # Q / Esc — quit
            print("[INFO] Quitting...")
            drone_ctrl.emergency_stop()
            break

        elif key == ord(' '):                  # SPACE — takeoff / land
            if not state.airborne:
                if state.drone_mode and not drone_ctrl.active:
                    drone_ctrl.connect()
                drone_ctrl.takeoff()
            else:
                drone_ctrl.land()

        elif key == ord('s'):                  # S — start tracking
            state.set(tracking=True, status_msg="Tracking")
            print("[INFO] Tracking started")

        elif key == ord('p'):                  # P — pause tracking
            state.set(tracking=False, status_msg="Paused")
            if drone_ctrl.active and state.airborne:
                drone_ctrl.send_tracking_command(0)
            print("[INFO] Tracking paused")

        elif key == ord('r'):                  # R — reset
            state.set(tracking=False, person_x=0.5, person_y=0.5,
                      drone_x=0.5, drone_y=0.18, status_msg="Reset")
            print("[INFO] Reset")

        elif key == ord('d'):                  # D — toggle drone mode
            if not state.drone_mode:
                print("[INFO] Drone mode ON — connecting to Tello Wi-Fi...")
                state.set(drone_mode=True)
                success = drone_ctrl.connect()
                if not success:
                    state.set(drone_mode=False)
                    print("[INFO] Reverted to simulation mode")
            else:
                state.set(drone_mode=False)
                print("[INFO] Drone mode OFF — simulation only")

    # ── Cleanup ──────────────────────────────────────────────
    print("[INFO] Cleaning up...")
    drone_ctrl.disconnect()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done. Goodbye!")


# ════════════════════════════════════════════════
#  HSV COLOR TUNER  (run separately to find your color)
#  Usage: python drone_umbrella.py tune
# ════════════════════════════════════════════════
def hsv_tuner():
    """
    Interactive HSV tuner — shows sliders to find the right
    color range for your shirt/object.
    Copy the printed values into COLOR_LOWER / COLOR_UPPER above.
    """
    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    cv2.namedWindow("HSV Tuner")
    cv2.namedWindow("Mask")

    cv2.createTrackbar("H Low",  "HSV Tuner", 35,  179, lambda x: None)
    cv2.createTrackbar("H High", "HSV Tuner", 85,  179, lambda x: None)
    cv2.createTrackbar("S Low",  "HSV Tuner", 80,  255, lambda x: None)
    cv2.createTrackbar("S High", "HSV Tuner", 255, 255, lambda x: None)
    cv2.createTrackbar("V Low",  "HSV Tuner", 80,  255, lambda x: None)
    cv2.createTrackbar("V High", "HSV Tuner", 255, 255, lambda x: None)

    print("[TUNER] Move sliders until your object shows white in the Mask window.")
    print("[TUNER] Press Q when done — values will be printed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hl = cv2.getTrackbarPos("H Low",  "HSV Tuner")
        hh = cv2.getTrackbarPos("H High", "HSV Tuner")
        sl = cv2.getTrackbarPos("S Low",  "HSV Tuner")
        sh = cv2.getTrackbarPos("S High", "HSV Tuner")
        vl = cv2.getTrackbarPos("V Low",  "HSV Tuner")
        vh = cv2.getTrackbarPos("V High", "HSV Tuner")

        lower = np.array([hl, sl, vl])
        upper = np.array([hh, sh, vh])
        mask  = cv2.inRange(hsv, lower, upper)

        put_text(frame, f"Lower: {lower}", (10, 30), CLR_TEXT)
        put_text(frame, f"Upper: {upper}", (10, 55), CLR_TEXT)

        cv2.imshow("HSV Tuner", frame)
        cv2.imshow("Mask",      mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\n[TUNER] Copy these into your script:")
            print(f"COLOR_LOWER = np.array([{hl}, {sl}, {vl}])")
            print(f"COLOR_UPPER = np.array([{hh}, {sh}, {vh}])")
            break

    cap.release()
    cv2.destroyAllWindows()


# ════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        hsv_tuner()
    else:
        main()
