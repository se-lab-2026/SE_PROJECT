"""
Autonomous Drone Umbrella System — GUI with Face Detection
==========================================================

Overview:
This system uses computer vision to detect a person's face in real time and
moves the drone accordingly. It supports:
- Left / Right movement using face position
- Forward / Backward movement using face size (distance estimate)
- Live webcam feed
- Top-down simulation
- GUI controls for demo and testing
"""

import cv2
import numpy as np
import threading
import time
import tkinter as tk
import logging
from PIL import Image, ImageTk

logging.getLogger("djitellopy").setLevel(logging.WARNING)

try:
    from djitellopy import Tello
    TELLO_AVAILABLE = True
except ImportError:
    TELLO_AVAILABLE = False


# ════════════════════════════════════════════════
# SETTINGS
# ════════════════════════════════════════════════
WEBCAM_INDEX = 0
FRAME_W = 480
FRAME_H = 360
SIM_SIZE = 360

SPEED_LR = 35
SPEED_FB = 25
SPEED_UD = 0
SPEED_YAW = 0

RC_INTERVAL = 0.05   # Send RC commands at 20Hz (Tello's max reliable rate)

# TARGET_AREA: face pixel area at ideal hovering distance.
# From screenshot, face area = 17424 at normal distance.
TARGET_AREA = 17000
FB_DEADZONE = 0.20   # Large deadzone so FB doesn't fire unless clearly too close/far
FB_MIN_SPEED = 15
LR_DEADZONE = 0.05

# Laptop camera faces YOU, drone also faces YOU at takeoff.
# So drone's right = YOUR left on screen → we must invert.
# If drone still goes wrong way, flip this to False.
LR_INVERT = True

CLR_BOX = (50, 220, 100)
CLR_DRONE = (60, 60, 220)
CLR_PERSON = (220, 120, 50)
CLR_LINE = (180, 180, 180)
CLR_GRID = (230, 230, 228)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ════════════════════════════════════════════════
# STATE
# ════════════════════════════════════════════════
class AppState:
    def __init__(self):
        self.running = True
        self.tracking = False
        self.airborne = False
        self.drone_mode = False
        self.detected = False

        self.person_x = 0.5
        self.person_y = 0.5
        self.drone_x = 0.5
        self.drone_y = 0.18

        self.battery = 0
        self.fps = 0
        self.error_x = 0.0
        self.error_z = 0.0
        self.area = 0
        self.depth_state = "CENTERED"
        self.status = "Ready"
        self.sensitivity = 0.30

        self._lock = threading.Lock()

    def set(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)


# ════════════════════════════════════════════════
# DRONE
# ════════════════════════════════════════════════
class DroneController:
    def __init__(self, state):
        self.state = state
        self.tello = None
        self.active = False
        # Current RC values — updated by cam loop, sent by RC thread at 20Hz
        self._lr = 0
        self._fb = 0
        self._ud = 0
        self._yaw = 0
        self._rc_lock = threading.Lock()
        self._rc_thread = None

    def _rc_loop(self):
        """Dedicated thread: sends RC commands to Tello at exactly 20Hz."""
        while self.active and self.state.airborne:
            with self._rc_lock:
                lr, fb, ud, yaw = self._lr, self._fb, self._ud, self._yaw
            try:
                # lr = left/right, fb = forward/back relative to drone's nose direction
                self.tello.send_rc_control(lr, fb, ud, yaw)
            except Exception as e:
                print(f"[DRONE] RC error: {e}")
            time.sleep(RC_INTERVAL)

    def set_rc(self, lr, fb, ud=0, yaw=0):
        """Update desired RC values. The RC thread picks these up at 20Hz."""
        with self._rc_lock:
            self._lr = int(lr)
            self._fb = int(fb)
            self._ud = int(ud)
            self._yaw = int(yaw)

    def connect(self):
        if not TELLO_AVAILABLE:
            return False, "djitellopy not installed. Run: pip install djitellopy"
        try:
            print("[DRONE] Connecting to Tello...")
            self.tello = Tello()
            self.tello.connect()
            bat = self.tello.get_battery()
            self.active = True
            self.state.set(battery=bat)
            print(f"[DRONE] Connected! Battery: {bat}%")
            return True, f"Connected! Battery: {bat}%"
        except Exception as e:
            self.active = False
            print(f"[DRONE] Connection failed: {e}")
            return False, str(e)

    def takeoff(self):
        if self.active and self.tello:
            try:
                print("[DRONE] Taking off...")
                self.tello.takeoff()
                time.sleep(1)
                try:
                    self.state.set(battery=self.tello.get_battery())
                except Exception:
                    pass
                self.state.set(airborne=True, status="Airborne")
                # Start the dedicated RC thread now that we're airborne
                self._rc_thread = threading.Thread(target=self._rc_loop, daemon=True)
                self._rc_thread.start()
            except Exception as e:
                print(f"[DRONE] Takeoff error: {e}")
        else:
            self.state.set(airborne=True, status="Airborne (sim)")

    def land(self):
        if self.active and self.tello:
            try:
                print("[DRONE] Landing...")
                self.set_rc(0, 0, 0, 0)
                time.sleep(0.1)
                self.tello.land()
            except Exception as e:
                print(f"[DRONE] Land error: {e}")
        self.state.set(airborne=False, status="Landed")

    def stop_motion(self):
        self.set_rc(0, 0, 0, 0)

    def send_rc(self, lr, fb, ud=0, yaw=0):
        """Kept for compatibility — just updates the RC values."""
        self.set_rc(lr, fb, ud, yaw)

    def disconnect(self):
        if self.active and self.tello:
            try:
                self.set_rc(0, 0, 0, 0)
                time.sleep(0.1)
                if self.state.airborne:
                    self.tello.land()
                    time.sleep(1)
                self.tello.end()
            except Exception:
                pass
        self.active = False


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
        return x + w // 2, y + h // 2, (x, y, w, h), w * h
    return None, None, None, 0


# ════════════════════════════════════════════════
# CONTROL HELPERS
# ════════════════════════════════════════════════
def apply_deadzone(value, deadzone):
    if abs(value) < deadzone:
        return 0.0
    return value


def scaled_fb_speed(error_z):
    ez = apply_deadzone(error_z, FB_DEADZONE)
    if ez == 0:
        return 0

    raw = int(np.clip(ez * SPEED_FB, -SPEED_FB, SPEED_FB))

    if raw > 0:
        return max(raw, FB_MIN_SPEED)
    return min(raw, -FB_MIN_SPEED)


def scaled_lr_speed(error_x):
    ex = apply_deadzone(error_x, LR_DEADZONE)
    if ex == 0:
        return 0
    return int(np.clip(ex * SPEED_LR, -SPEED_LR, SPEED_LR))


# ════════════════════════════════════════════════
# SIMULATION
# ════════════════════════════════════════════════
def make_sim(state):
    s = SIM_SIZE
    img = np.full((s, s, 3), 248, dtype=np.uint8)

    step = s // 10
    for i in range(0, s, step):
        cv2.line(img, (i, 0), (i, s), CLR_GRID, 1)
        cv2.line(img, (0, i), (s, i), CLR_GRID, 1)

    cv2.line(img, (s // 2, 0), (s // 2, s), (210, 210, 210), 1)
    cv2.line(img, (0, s // 2), (s, s // 2), (210, 210, 210), 1)

    px = int(state.person_x * s)
    py = int(state.person_y * s)
    dx = int(state.drone_x * s)
    dy = int(state.drone_y * s)

    cv2.line(img, (dx, dy), (px, py), CLR_LINE, 2)

    cv2.ellipse(img, (px, py + 20), (16, 5), 0, 0, 360, (210, 210, 210), -1)
    cv2.circle(img, (px, py), 16, CLR_PERSON, -1)
    cv2.circle(img, (px, py), 16, (80, 50, 10), 2)
    cv2.putText(
        img, "P", (px - 6, py + 5), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 2, cv2.LINE_AA
    )

    for ang in [45, 135, 225, 315]:
        rad = np.radians(ang)
        ex = int(dx + 14 * np.cos(rad))
        ey = int(dy + 14 * np.sin(rad))
        cv2.line(img, (dx, dy), (ex, ey), (130, 130, 130), 2)
        cv2.circle(img, (ex, ey), 5, (160, 160, 160), -1)

    cv2.circle(img, (dx, dy), 9, CLR_DRONE, -1)
    cv2.circle(img, (dx, dy), 9, (10, 10, 120), 2)
    cv2.putText(
        img, "D", (dx - 5, dy + 4), cv2.FONT_HERSHEY_SIMPLEX,
        0.4, (255, 255, 255), 2, cv2.LINE_AA
    )

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ════════════════════════════════════════════════
# APP
# ════════════════════════════════════════════════
class DroneApp:
    def __init__(self, root):
        self.root = root
        self.state = AppState()
        self.drone = DroneController(self.state)

        self.smooth_x = 0.5
        self.smooth_y = 0.5
        self.cam_frame = None

        self._build_ui()
        self._start_camera()
        self._update_loop()

    def _build_ui(self):
        self.root.title("Autonomous Drone Umbrella System")
        self.root.configure(bg="#0f0f0f")
        self.root.resizable(False, False)

        tb = tk.Frame(self.root, bg="#0f0f0f", pady=10)
        tb.pack(fill="x", padx=16)

        tk.Label(
            tb,
            text="Autonomous Drone Umbrella",
            font=("Courier New", 16, "bold"),
            fg="#00e676",
            bg="#0f0f0f"
        ).pack(side="left")

        self.mode_badge = tk.Label(
            tb,
            text="  SIMULATION  ",
            font=("Courier New", 10),
            fg="#ffb300",
            bg="#1a1a1a",
            padx=6,
            pady=2
        )
        self.mode_badge.pack(side="right")

        tk.Label(
            self.root,
            text="Detection: Face Detection (Haar Cascade) — position + distance control",
            font=("Courier New", 9),
            fg="#444",
            bg="#0f0f0f"
        ).pack(anchor="w", padx=16, pady=(0, 6))

        content = tk.Frame(self.root, bg="#0f0f0f")
        content.pack(padx=16)

        left = tk.Frame(content, bg="#0f0f0f")
        left.pack(side="left", padx=(0, 12))
        tk.Label(left, text="WEBCAM FEED", font=("Courier New", 9), fg="#555", bg="#0f0f0f").pack(anchor="w")
        self.cam_label = tk.Label(left, bg="#1a1a1a", width=FRAME_W, height=FRAME_H)
        self.cam_label.pack()

        right = tk.Frame(content, bg="#0f0f0f")
        right.pack(side="left")
        tk.Label(right, text="SIMULATION VIEW", font=("Courier New", 9), fg="#555", bg="#0f0f0f").pack(anchor="w")
        self.sim_label = tk.Label(right, bg="#f8f8f8", width=SIM_SIZE, height=SIM_SIZE)
        self.sim_label.pack()

        sf = tk.Frame(self.root, bg="#141414")
        sf.pack(fill="x", padx=16, pady=8)

        self.lbl_det = self._stat(sf, "Detection", "NO", "#ef5350")
        self.lbl_trk = self._stat(sf, "Tracking", "PAUSED", "#ffb300")
        self.lbl_pos = self._stat(sf, "Person Pos", "(0.50, 0.50)", "#aaa")
        self.lbl_drn = self._stat(sf, "Drone Pos", "(0.50, 0.18)", "#aaa")
        self.lbl_errx = self._stat(sf, "Error X", "0.00", "#aaa")
        self.lbl_errz = self._stat(sf, "Error Z", "0.00", "#aaa")
        self.lbl_area = self._stat(sf, "Face Area", "0", "#aaa")
        self.lbl_depth = self._stat(sf, "Depth", "CENTERED", "#aaa")
        self.lbl_bat = self._stat(sf, "Battery", "0%", "#aaa")
        self.lbl_fps = self._stat(sf, "FPS", "0", "#aaa")
        self.lbl_stat = self._stat(sf, "Status", "Ready", "#00e676")

        bf = tk.Frame(self.root, bg="#0f0f0f")
        bf.pack(padx=16, pady=(0, 10), fill="x")

        self.btn_start = self._btn(bf, "START TRACKING", "#00e676", "#000", self._start)
        self.btn_pause = self._btn(bf, "PAUSE", "#ffb300", "#000", self._pause)
        self.btn_reset = self._btn(bf, "RESET", "#2196f3", "#fff", self._reset)
        self.btn_takeoff = self._btn(bf, "TAKEOFF", "#7c4dff", "#fff", self._takeoff_land)
        self.btn_drone = self._btn(bf, "CONNECT DRONE", "#1a1a1a", "#00e676", self._toggle_drone)
        self.btn_quit = self._btn(bf, "QUIT", "#ef5350", "#fff", self._quit)

        for b in [
            self.btn_start, self.btn_pause, self.btn_reset,
            self.btn_takeoff, self.btn_drone, self.btn_quit
        ]:
            b.pack(side="left", padx=4, pady=4)

        sf2 = tk.Frame(self.root, bg="#0f0f0f")
        sf2.pack(padx=16, pady=(0, 16), fill="x")

        tk.Label(
            sf2,
            text="Tracking sensitivity",
            font=("Courier New", 9),
            fg="#555",
            bg="#0f0f0f"
        ).pack(side="left", padx=(0, 10))

        self.sens_var = tk.DoubleVar(value=0.30)
        tk.Scale(
            sf2,
            from_=0.05,
            to=1.0,
            resolution=0.05,
            variable=self.sens_var,
            orient="horizontal",
            length=200,
            bg="#0f0f0f",
            fg="#aaa",
            troughcolor="#1a1a1a",
            highlightthickness=0,
            bd=0,
            command=lambda v: self.state.set(sensitivity=float(v))
        ).pack(side="left")

        self.sens_lbl = tk.Label(
            sf2,
            text="0.30",
            font=("Courier New", 9),
            fg="#aaa",
            bg="#0f0f0f"
        )
        self.sens_lbl.pack(side="left", padx=8)

    def _stat(self, p, key, val, color):
        f = tk.Frame(p, bg="#141414", padx=10, pady=8)
        f.pack(side="left", expand=True, fill="x", padx=3, pady=6)
        tk.Label(f, text=key.upper(), font=("Courier New", 8), fg="#444", bg="#141414").pack(anchor="w")
        lbl = tk.Label(f, text=val, font=("Courier New", 11, "bold"), fg=color, bg="#141414")
        lbl.pack(anchor="w")
        return lbl

    def _btn(self, p, text, bg, fg, cmd):
        return tk.Button(
            p,
            text=text,
            font=("Courier New", 9, "bold"),
            bg=bg,
            fg=fg,
            relief="flat",
            padx=12,
            pady=6,
            cursor="hand2",
            activebackground=bg,
            command=cmd
        )

    def _start(self):
        self.state.set(tracking=True, status="Tracking")
        self.lbl_trk.config(text="ACTIVE", fg="#00e676")

    def _pause(self):
        self.state.set(tracking=False, status="Paused")
        self.lbl_trk.config(text="PAUSED", fg="#ffb300")
        self.drone.stop_motion()

    def _reset(self):
        self.state.set(
            tracking=False,
            person_x=0.5,
            person_y=0.5,
            drone_x=0.5,
            drone_y=0.18,
            error_x=0.0,
            error_z=0.0,
            area=0,
            depth_state="CENTERED",
            status="Reset"
        )
        self.smooth_x = 0.5
        self.smooth_y = 0.5
        self.lbl_trk.config(text="PAUSED", fg="#ffb300")
        self.drone.stop_motion()

    def _takeoff_land(self):
        if not self.state.drone_mode:
            self.lbl_stat.config(text="Connect drone first!", fg="#ef5350")
            return

        if not self.state.airborne:
            self.drone.takeoff()
            self.btn_takeoff.config(text="LAND", bg="#ef5350")
            # Auto-start tracking once airborne so drone moves immediately
            self.state.set(tracking=True, status="Tracking")
            self.lbl_trk.config(text="ACTIVE", fg="#00e676")
        else:
            self.drone.land()
            self.state.set(tracking=False)
            self.btn_takeoff.config(text="TAKEOFF", bg="#7c4dff")
            self.lbl_trk.config(text="PAUSED", fg="#ffb300")

    def _toggle_drone(self):
        if not self.state.drone_mode:
            self.lbl_stat.config(text="Connecting...", fg="#ffb300")
            self.root.update()
            ok, msg = self.drone.connect()
            if ok:
                self.state.set(drone_mode=True, status=msg)
                self.btn_drone.config(text="DISCONNECT", fg="#ef5350")
                self.mode_badge.config(text="  DRONE  ", fg="#ef5350")
                self.lbl_stat.config(text=msg, fg="#00e676")
            else:
                self.state.set(status=f"Failed: {msg}")
                self.lbl_stat.config(text=f"Failed: {msg}", fg="#ef5350")
        else:
            self.drone.disconnect()
            self.state.set(drone_mode=False, airborne=False, status="Disconnected")
            self.btn_drone.config(text="CONNECT DRONE", fg="#00e676")
            self.btn_takeoff.config(text="TAKEOFF", bg="#7c4dff")
            self.mode_badge.config(text="  SIMULATION  ", fg="#ffb300")

    def _quit(self):
        self.state.set(running=False)
        self.drone.disconnect()
        self.root.after(300, self.root.destroy)

    def _start_camera(self):
        self.cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

        self.fps_timer = time.time()
        self.fps_count = 0

        threading.Thread(target=self._cam_loop, daemon=True).start()

    def _cam_loop(self):
        error_x = 0.0
        error_z = 0.0

        while self.state.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]

            cx, cy, bbox, area = detect_face(frame)
            detected = cx is not None

            if detected:
                nx = cx / fw
                ny = cy / fh
                sens = self.state.sensitivity

                self.smooth_x = self.smooth_x * (1 - sens) + nx * sens
                self.smooth_y = self.smooth_y * (1 - sens) + ny * sens

                # LR direction: positive error_x = drone goes right
                # If drone goes wrong direction, set LR_INVERT = True at top of file
                raw_error_x = (self.smooth_x - 0.5) * 2.0
                error_x = -raw_error_x if LR_INVERT else raw_error_x
                error_z = (TARGET_AREA - area) / TARGET_AREA
                error_z = float(np.clip(error_z, -1.5, 1.5))

                if error_z > FB_DEADZONE:
                    depth_state = "TOO FAR"
                elif error_z < -FB_DEADZONE:
                    depth_state = "TOO CLOSE"
                else:
                    depth_state = "CENTERED"

                self.state.set(
                    person_x=self.smooth_x,
                    person_y=self.smooth_y,
                    detected=True,
                    error_x=error_x,
                    error_z=error_z,
                    area=area,
                    depth_state=depth_state,
                    status="Tracking"
                )

                if bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), CLR_BOX, 2)

                    L = 15
                    for (x1, y1, x2, y2) in [
                        (x, y, x + L, y), (x, y, x, y + L),
                        (x + w, y, x + w - L, y), (x + w, y, x + w, y + L),
                        (x, y + h, x + L, y + h), (x, y + h, x, y + h - L),
                        (x + w, y + h, x + w - L, y + h), (x + w, y + h, x + w, y + h - L)
                    ]:
                        cv2.line(frame, (x1, y1), (x2, y2), CLR_BOX, 3)

                    cv2.circle(frame, (cx, cy), 4, CLR_BOX, -1)
                    cv2.putText(
                        frame,
                        f"FACE ({cx},{cy}) A={area}",
                        (x, max(y - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, CLR_BOX, 1, cv2.LINE_AA
                    )

                # Send RC whenever tracking is active AND drone is airborne
                # Also send when airborne but not tracking (keep-alive zeros)
                if self.state.airborne:
                    if self.state.tracking:
                        lr = scaled_lr_speed(error_x)
                        fb = scaled_fb_speed(error_z)
                        print(f"[RC] lr={lr:+d} fb={fb:+d}  ex={error_x:.3f} ez={error_z:.3f}")
                        self.drone.send_rc(lr, fb, SPEED_UD, SPEED_YAW)
                    else:
                        # Keep-alive: drone hovers, won't auto-land
                        self.drone.send_rc(0, 0, 0, 0)

            else:
                error_x = 0.0
                error_z = 0.0
                self.state.set(
                    detected=False,
                    error_x=0.0,
                    error_z=0.0,
                    area=0,
                    depth_state="NO FACE",
                    status="Searching..." if self.state.tracking else "Paused"
                )
                # Keep-alive so Tello doesn't auto-land after 1s with no RC
                if self.state.airborne:
                    self.drone.send_rc(0, 0, 0, 0)

            # Simulation top-down view update
            self.state.drone_x = self.state.drone_x * 0.88 + self.state.person_x * 0.12
            target_drone_y = self.state.person_y - 0.15 - (error_z * 0.10)
            self.state.drone_y = float(np.clip(
                self.state.drone_y * 0.88 + target_drone_y * 0.12,
                0.05, 0.95
            ))

            self.fps_count += 1
            if time.time() - self.fps_timer >= 1.0:
                self.state.set(fps=self.fps_count)
                self.fps_count = 0
                self.fps_timer = time.time()

            self.cam_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            time.sleep(0.01)

    def _update_loop(self):
        if not self.state.running:
            return

        s = self.state

        if self.cam_frame is not None:
            img = Image.fromarray(self.cam_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imgtk
            self.cam_label.config(image=imgtk)

        sim_arr = make_sim(s)
        sim_img = Image.fromarray(sim_arr)
        sim_tk = ImageTk.PhotoImage(image=sim_img)
        self.sim_label.imgtk = sim_tk
        self.sim_label.config(image=sim_tk)

        self.lbl_det.config(text="YES" if s.detected else "NO", fg="#00e676" if s.detected else "#ef5350")
        self.lbl_pos.config(text=f"({s.person_x:.2f}, {s.person_y:.2f})")
        self.lbl_drn.config(text=f"({s.drone_x:.2f}, {s.drone_y:.2f})")
        self.lbl_errx.config(text=f"{s.error_x:+.2f}")
        self.lbl_errz.config(text=f"{s.error_z:+.2f}")
        self.lbl_area.config(text=str(s.area))
        self.lbl_depth.config(
            text=s.depth_state,
            fg="#ffb300" if s.depth_state in ("TOO FAR", "TOO CLOSE") else "#aaa"
        )
        self.lbl_bat.config(text=f"{s.battery}%")
        self.lbl_fps.config(text=str(s.fps))
        self.lbl_stat.config(text=s.status)
        self.sens_lbl.config(text=f"{self.sens_var.get():.2f}")

        self.root.after(33, self._update_loop)


# ════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = DroneApp(root)
    root.protocol("WM_DELETE_WINDOW", app._quit)
    root.mainloop()

    if hasattr(app, "cap") and app.cap.isOpened():
        app.cap.release()

    print("Goodbye!")