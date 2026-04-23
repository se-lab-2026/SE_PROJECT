# ------------------------------------------------------------
# Controls (Keyboard)
# ------------------------------------------------------------
# D       → Toggle Drone Mode ON/OFF (Connect to Tello)
# SPACE   → Takeoff / Land
# S       → Start Tracking
# P       → Pause Tracking
# R       → Reset Simulation
# Q       → Emergency Stop and Quit
# ------------------------------------------------------------

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
    print("[INFO] Install with: pip install djitellopy")


# ════════════════════════════════════════════════
# SETTINGS
# ════════════════════════════════════════════════
WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SIM_SIZE = 500

SPEED_LR = 30
SPEED_FB = 55
SPEED_UD = 0
SPEED_YAW = 0

TARGET_AREA = 15000
FB_DEADZONE = 0.02
FB_MIN_SPEED = 22
LR_DEADZONE = 0.05

PERSON_SMOOTH = 0.4
DRONE_SMOOTH = 0.15

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

CLR_DRONE = (50, 50, 220)
CLR_PERSON = (220, 120, 50)
CLR_LINE = (160, 160, 160)
CLR_BOX = (50, 220, 100)
CLR_TEXT = (240, 240, 240)
CLR_PANEL = (30, 30, 30)
CLR_GRID = (60, 60, 60)
CLR_TRACKING = (50, 200, 100)
CLR_LOST = (50, 80, 220)
CLR_WARN = (30, 160, 220)


# ════════════════════════════════════════════════
# STATE
# ════════════════════════════════════════════════
class AppState:
    def __init__(self):
        self.tracking = False
        self.airborne = False
        self.drone_mode = False
        self.running = True

        self.person_x = 0.5
        self.person_y = 0.5
        self.drone_x = 0.5
        self.drone_y = 0.2

        self.battery = 0
        self.detected = False
        self.fps = 0
        self.error_x = 0.0
        self.error_z = 0.0
        self.area = 0
        self.depth_state = "CENTERED"
        self.status_msg = "Ready"

        self._lock = threading.Lock()

    def set(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)


# ════════════════════════════════════════════════
# CONTROL HELPERS
# ════════════════════════════════════════════════
def apply_deadzone(value, deadzone):
    if abs(value) < deadzone:
        return 0.0
    return value


def scaled_lr_speed(error_x: float) -> int:
    ex = apply_deadzone(error_x, LR_DEADZONE)
    if ex == 0:
        return 0
    return int(np.clip(ex * SPEED_LR, -SPEED_LR, SPEED_LR))


def scaled_fb_speed(error_z: float) -> int:
    ez = apply_deadzone(error_z, FB_DEADZONE)
    if ez == 0:
        return 0

    raw = int(np.clip(ez * SPEED_FB, -SPEED_FB, SPEED_FB))

    if raw > 0:
        return max(raw, FB_MIN_SPEED)
    return min(raw, -FB_MIN_SPEED)


# ════════════════════════════════════════════════
# DRONE CONTROLLER
# ════════════════════════════════════════════════
class DroneController:
    def __init__(self, state: AppState):
        self.state = state
        self.tello = None
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
            self.active = False
            return False

    def takeoff(self):
        if self.active and self.tello:
            try:
                print("[DRONE] Taking off...")
                self.tello.takeoff()
                time.sleep(1)
                self.state.set(airborne=True, status_msg="Airborne")
                try:
                    self.state.set(battery=self.tello.get_battery())
                except Exception:
                    pass
            except Exception as e:
                print(f"[DRONE] Takeoff error: {e}")
        else:
            self.state.set(airborne=True, status_msg="Airborne (sim)")

    def land(self):
        if self.active and self.tello:
            try:
                print("[DRONE] Landing...")
                self.tello.send_rc_control(0, 0, 0, 0)
                self.tello.land()
                self.state.set(airborne=False, status_msg="Landed")
            except Exception as e:
                print(f"[DRONE] Land error: {e}")
        else:
            self.state.set(airborne=False, status_msg="Landed (sim)")

    def emergency_stop(self):
        try:
            if self.active and self.tello:
                self.tello.send_rc_control(0, 0, 0, 0)
                if self.state.airborne:
                    self.tello.land()
        except Exception:
            pass
        self.state.set(airborne=False, running=False)

    def send_tracking_command(self, error_x: float, error_z: float):
        lr = scaled_lr_speed(error_x)
        fb = scaled_fb_speed(error_z)

        if self.active and self.tello and self.state.airborne:
            try:
                self.tello.send_rc_control(lr, fb, SPEED_UD, SPEED_YAW)
            except Exception as e:
                print(f"[DRONE] RC error: {e}")

    def stop_motion(self):
        if self.active and self.tello and self.state.airborne:
            try:
                self.tello.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass

    def disconnect(self):
        if self.active and self.tello:
            try:
                self.stop_motion()
                if self.state.airborne:
                    self.tello.land()
                self.tello.end()
            except Exception:
                pass


# ════════════════════════════════════════════════
# DETECTION
# ════════════════════════════════════════════════
def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        return cx, cy, area, (x, y, w, h)

    return None, None, 0, None


# ════════════════════════════════════════════════
# DRAWING HELPERS
# ════════════════════════════════════════════════
def draw_panel(img, x, y, w, h, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), CLR_PANEL, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)


def put_text(img, text, pos, color=CLR_TEXT, scale=0.55, thickness=1):
    cv2.putText(
        img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA
    )


def draw_status_overlay(frame, state: AppState):
    h, w = frame.shape[:2]

    draw_panel(frame, 0, 0, w, 75)

    tracking_color = CLR_TRACKING if state.detected else CLR_LOST
    tracking_label = "TRACKING" if state.detected else "LOST"

    put_text(frame, "Autonomous Drone Umbrella", (10, 20), CLR_TEXT, 0.6, 1)
    put_text(frame, f"FPS: {state.fps}", (10, 45), CLR_TEXT, 0.45)
    put_text(frame, f"State: {tracking_label}", (110, 45), tracking_color, 0.45)
    put_text(frame, f"Mode: {'DRONE' if state.drone_mode else 'SIM'}", (230, 45),
             CLR_WARN if state.drone_mode else CLR_TEXT, 0.45)
    put_text(frame, f"Battery: {state.battery}%", (340, 45), CLR_TEXT, 0.45)
    if state.airborne:
        put_text(frame, "AIRBORNE", (530, 20), CLR_WARN, 0.55, 2)

    draw_panel(frame, 0, h - 40, w, 40)
    put_text(frame, "SPACE:Takeoff/Land  S:Track  P:Pause  R:Reset  D:Drone Mode  Q:Quit",
             (8, h - 15), CLR_TEXT, 0.4)

    if state.detected:
        put_text(frame, f"X Error: {state.error_x:+.2f}", (10, h - 55), CLR_TRACKING, 0.45)
        put_text(frame, f"Z Error: {state.error_z:+.2f}", (150, h - 55), CLR_WARN, 0.45)
        put_text(frame, f"Area: {state.area}", (290, h - 55), CLR_TEXT, 0.45)
        put_text(frame, f"Depth: {state.depth_state}", (400, h - 55), CLR_TEXT, 0.45)


def draw_simulation(state: AppState):
    sim = np.ones((SIM_SIZE, SIM_SIZE, 3), dtype=np.uint8) * 245
    sim[:] = (245, 245, 242)

    step = SIM_SIZE // 10
    for i in range(0, SIM_SIZE, step):
        cv2.line(sim, (i, 0), (i, SIM_SIZE), CLR_GRID, 1)
        cv2.line(sim, (0, i), (SIM_SIZE, i), CLR_GRID, 1)

    cx = SIM_SIZE // 2
    cv2.line(sim, (cx, 0), (cx, SIM_SIZE), (180, 180, 180), 1)
    cv2.line(sim, (0, cx), (SIM_SIZE, cx), (180, 180, 180), 1)

    px = int(state.person_x * SIM_SIZE)
    py = int(state.person_y * SIM_SIZE)
    dx = int(state.drone_x * SIM_SIZE)
    dy = int(state.drone_y * SIM_SIZE)

    cv2.line(sim, (dx, dy), (px, py), CLR_LINE, 2)

    cv2.ellipse(sim, (px, py + 22), (18, 6), 0, 0, 360, (200, 200, 200), -1)
    cv2.circle(sim, (px, py), 18, CLR_PERSON, -1)
    cv2.circle(sim, (px, py), 18, (80, 60, 20), 2)
    put_text(sim, "P", (px - 5, py + 5), (255, 255, 255), 0.55, 2)

    for angle in [45, 135, 225, 315]:
        rad = np.radians(angle)
        ex = int(dx + 16 * np.cos(rad))
        ey = int(dy + 16 * np.sin(rad))
        cv2.line(sim, (dx, dy), (ex, ey), (100, 100, 100), 2)
        cv2.circle(sim, (ex, ey), 5, (140, 140, 140), -1)

    cv2.circle(sim, (dx, dy), 10, CLR_DRONE, -1)
    cv2.circle(sim, (dx, dy), 10, (10, 10, 120), 2)
    put_text(sim, "D", (dx - 5, dy + 5), (255, 255, 255), 0.45, 2)

    draw_panel(sim, 0, SIM_SIZE - 95, SIM_SIZE, 95, alpha=0.75)
    put_text(sim, f"Tracking: {'ACTIVE' if state.tracking else 'PAUSED'}",
             (10, SIM_SIZE - 72), CLR_TEXT, 0.45)
    put_text(sim, f"Detection: {'YES' if state.detected else 'NO'}",
             (10, SIM_SIZE - 52), CLR_TRACKING if state.detected else CLR_LOST, 0.45)
    put_text(sim, f"Person: ({state.person_x:.2f}, {state.person_y:.2f})",
             (10, SIM_SIZE - 32), CLR_TEXT, 0.45)
    put_text(sim, f"Drone: ({state.drone_x:.2f}, {state.drone_y:.2f})",
             (10, SIM_SIZE - 12), CLR_TEXT, 0.45)
    put_text(sim, f"Status: {state.status_msg}",
             (250, SIM_SIZE - 52), CLR_WARN, 0.45)
    put_text(sim, f"X Err: {state.error_x:+.2f}",
             (250, SIM_SIZE - 32), CLR_TEXT, 0.45)
    put_text(sim, f"Z Err: {state.error_z:+.2f}",
             (250, SIM_SIZE - 12), CLR_TEXT, 0.45)

    draw_panel(sim, 0, 0, SIM_SIZE, 30, alpha=0.7)
    put_text(sim, "Drone Umbrella — Top-down simulation", (10, 20), CLR_TEXT, 0.48)

    return sim


# ════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Autonomous Drone Umbrella System — Face Detection")
    print("=" * 60)
    print("Controls:")
    print("  D      -> Toggle drone mode / connect to Tello")
    print("  SPACE  -> Takeoff / Land")
    print("  S      -> Start tracking")
    print("  P      -> Pause tracking")
    print("  R      -> Reset simulation")
    print("  Q      -> Quit")
    print("=" * 60)

    state = AppState()
    drone_ctrl = DroneController(state)

    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam index {WEBCAM_INDEX}. Try 1 instead.")
        return

    print("[OK] Webcam opened.")

    cv2.namedWindow("Webcam Feed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Simulation View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam Feed", FRAME_WIDTH, FRAME_HEIGHT)
    cv2.resizeWindow("Simulation View", SIM_SIZE, SIM_SIZE)
    cv2.moveWindow("Webcam Feed", 0, 0)
    cv2.moveWindow("Simulation View", FRAME_WIDTH + 20, 0)

    fps_timer = time.time()
    fps_count = 0
    smooth_x = 0.5
    smooth_y = 0.5

    while state.running:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        cx, cy, area, bbox = detect_face(frame)
        detected = cx is not None

        if detected:
            nx = cx / fw
            ny = cy / fh

            smooth_x = smooth_x * (1 - PERSON_SMOOTH) + nx * PERSON_SMOOTH
            smooth_y = smooth_y * (1 - PERSON_SMOOTH) + ny * PERSON_SMOOTH

            error_x = (smooth_x - 0.5) * 2.0
            error_z = (TARGET_AREA - area) / TARGET_AREA
            error_z = float(np.clip(error_z, -1.5, 1.5))

            print(f"[CORE] AREA={area} TARGET={TARGET_AREA} ERROR_Z={error_z:.3f}")

            if error_z > FB_DEADZONE:
                depth_state = "TOO FAR"
            elif error_z < -FB_DEADZONE:
                depth_state = "TOO CLOSE"
            else:
                depth_state = "CENTERED"

            state.set(
                person_x=smooth_x,
                person_y=smooth_y,
                detected=True,
                error_x=error_x,
                error_z=error_z,
                area=area,
                depth_state=depth_state,
                status_msg="Tracking"
            )

            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), CLR_BOX, 2)
                cv2.circle(frame, (cx, cy), 6, CLR_BOX, -1)
                put_text(frame, f"Face Area: {area}", (x, max(y - 8, 20)), CLR_BOX, 0.45)

            if state.tracking:
                lr = scaled_lr_speed(error_x)
                fb = scaled_fb_speed(error_z)
                print(f"[RC] lr={lr} fb={fb}  error_x={error_x:.3f} error_z={error_z:.3f}")
                # Send to real drone only when airborne
                if state.airborne:
                    drone_ctrl.send_tracking_command(error_x, error_z)
            elif state.airborne:
                drone_ctrl.stop_motion()

        else:
            state.set(
                detected=False,
                error_x=0.0,
                error_z=0.0,
                area=0,
                depth_state="NO FACE",
                status_msg="Searching..." if state.tracking else "Paused"
            )
            if state.tracking and state.airborne:
                drone_ctrl.stop_motion()

        # Simulation: drone smoothly follows person left/right (X axis)
        state.drone_x = state.drone_x * (1 - DRONE_SMOOTH) + state.person_x * DRONE_SMOOTH
        # Simulation top-down view: drone stays ABOVE person (lower Y value = top of screen)
        # error_z > 0 → person is far → drone moves toward person (closes the gap from above)
        # error_z < 0 → person too close → drone backs away (moves further above)
        target_drone_y = state.person_y - 0.15 - (state.error_z * 0.10)
        state.drone_y = float(np.clip(
            state.drone_y * (1 - DRONE_SMOOTH) + target_drone_y * DRONE_SMOOTH,
            0.05, 0.95
        ))

        fps_count += 1
        if time.time() - fps_timer >= 1.0:
            state.set(fps=fps_count)
            fps_count = 0
            fps_timer = time.time()

        draw_status_overlay(frame, state)
        sim = draw_simulation(state)

        cv2.imshow("Webcam Feed", frame)
        cv2.imshow("Simulation View", sim)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            print("[INFO] Quitting...")
            drone_ctrl.emergency_stop()
            break

        elif key == ord(' '):
            if not state.airborne:
                if state.drone_mode and not drone_ctrl.active:
                    success = drone_ctrl.connect()
                    if not success:
                        print("[INFO] Could not connect. Staying in simulation mode.")
                        state.set(drone_mode=False)
                drone_ctrl.takeoff()
            else:
                drone_ctrl.land()

        elif key == ord('s'):
            state.set(tracking=True, status_msg="Tracking")
            print("[INFO] Tracking started")

        elif key == ord('p'):
            state.set(tracking=False, status_msg="Paused")
            drone_ctrl.stop_motion()
            print("[INFO] Tracking paused")

        elif key == ord('r'):
            state.set(
                tracking=False,
                person_x=0.5,
                person_y=0.5,
                drone_x=0.5,
                drone_y=0.18,
                error_x=0.0,
                error_z=0.0,
                area=0,
                depth_state="CENTERED",
                status_msg="Reset"
            )
            smooth_x = 0.5
            smooth_y = 0.5
            drone_ctrl.stop_motion()
            print("[INFO] Reset")

        elif key == ord('d'):
            if not state.drone_mode:
                print("[INFO] Drone mode ON — connecting...")
                success = drone_ctrl.connect()
                if success:
                    state.set(drone_mode=True, status_msg="Drone connected")
                else:
                    state.set(drone_mode=False, status_msg="Simulation only")
                    print("[INFO] Reverted to simulation mode")
            else:
                print("[INFO] Drone mode OFF — simulation only")
                drone_ctrl.disconnect()
                state.set(drone_mode=False, airborne=False, status_msg="Simulation only")

    print("[INFO] Cleaning up...")
    drone_ctrl.disconnect()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done. Goodbye!")


if __name__ == "__main__":
    main()