"""
Autonomous Drone Umbrella System — GUI with Face Detection
===========================================================
No colored object needed — detects your face automatically!

Requirements:
    pip install opencv-python djitellopy numpy Pillow

Run:
    python drone_umbrella_gui_face.py
"""

import cv2
import numpy as np
import threading
import time
import tkinter as tk
import logging
import socket
from PIL import Image, ImageTk

logging.getLogger('djitellopy').setLevel(logging.WARNING)

try:
    from djitellopy import Tello
    TELLO_AVAILABLE = True
except ImportError:
    TELLO_AVAILABLE = False

# ════════════════════════════════════════════════
#  SETTINGS
# ════════════════════════════════════════════════
WEBCAM_INDEX  = 0
FRAME_W       = 480
FRAME_H       = 360
SIM_SIZE      = 360
SPEED_LR      = 20

CLR_BOX    = (50,  220, 100)
CLR_DRONE  = (60,   60, 220)
CLR_PERSON = (220, 120,  50)
CLR_LINE   = (180, 180, 180)
CLR_GRID   = (230, 230, 228)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ════════════════════════════════════════════════
#  STATE
# ════════════════════════════════════════════════
class AppState:
    def __init__(self):
        self.running    = True
        self.tracking   = False
        self.airborne   = False
        self.drone_mode = False
        self.detected   = False
        self.person_x   = 0.5
        self.person_y   = 0.5
        self.drone_x    = 0.5
        self.drone_y    = 0.18
        self.battery    = 0
        self.fps        = 0
        self.error_x    = 0.0
        self.status     = "Ready"
        self.sensitivity= 0.3
        self._lock      = threading.Lock()

    def set(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)


# ════════════════════════════════════════════════
#  DRONE
# ════════════════════════════════════════════════
class DroneController:
    def __init__(self, state):
        self.state  = state
        self.sock   = None
        self.active = False
        self._lock  = threading.Lock()

    def _send(self, cmd):
        try:
            self.sock.sendto(cmd.encode(), ("192.168.10.1", 8889))
            try:
                resp, _ = self.sock.recvfrom(1024)
                return resp.decode("utf-8", errors="ignore").strip()
            except:
                return "ok"
        except Exception as e:
            print(f"[CMD ERROR] {e}")
            return "error"

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(5)
            self.sock.bind(("", 8889))

            # Flush the first corrupted response
            self.sock.sendto(b"command", ("192.168.10.1", 8889))
            try:
                self.sock.recvfrom(1024)
            except:
                pass
            time.sleep(0.5)

            # Send command again — this time gets clean ok
            self.sock.sendto(b"command", ("192.168.10.1", 8889))
            try:
                resp, _ = self.sock.recvfrom(1024)
                resp_str = resp.decode("utf-8", errors="ignore").strip()
            except:
                resp_str = "ok"

            if "ok" not in resp_str.lower():
                return False, f"No response from drone"

            # Get battery
            self.sock.sendto(b"battery?", ("192.168.10.1", 8889))
            try:
                bat_resp, _ = self.sock.recvfrom(1024)
                bat = int(bat_resp.decode("utf-8", errors="ignore").strip())
            except:
                bat = 75

            self.active = True
            self.state.set(battery=bat)
            return True, f"Connected! Battery: {bat}%"

        except Exception as e:
            self.active = False
            return False, str(e)

    def takeoff(self):
        if self.active:
            resp = self._send("takeoff")
            print(f"[TAKEOFF] {resp}")
            time.sleep(2)
            self.state.set(airborne=True, status="Airborne")
        else:
            self.state.set(airborne=True, status="Airborne (sim)")

    def land(self):
        if self.active:
            self._send("land")
        self.state.set(airborne=False, status="Landed")

    def send_rc(self, lr):
        if self.active and self.state.airborne:
            try:
                cmd = f"rc {lr} 0 20 0"
                self.sock.sendto(cmd.encode(), ("192.168.10.1", 8889))
            except:
                pass

    def disconnect(self):
        if self.active:
            try:
                self._send("land")
                time.sleep(1)
                self.sock.close()
            except:
                pass
        self.active = False


# ════════════════════════════════════════════════
#  DETECTION
# ════════════════════════════════════════════════
def detect_face(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        return x + w//2, y + h//2, (x, y, w, h)
    return None, None, None


# ════════════════════════════════════════════════
#  SIMULATION
# ════════════════════════════════════════════════
def make_sim(state):
    s   = SIM_SIZE
    img = np.full((s, s, 3), 248, dtype=np.uint8)
    step = s // 10
    for i in range(0, s, step):
        cv2.line(img, (i, 0), (i, s), CLR_GRID, 1)
        cv2.line(img, (0, i), (s, i), CLR_GRID, 1)
    cv2.line(img, (s//2, 0), (s//2, s), (210,210,210), 1)
    cv2.line(img, (0, s//2), (s, s//2), (210,210,210), 1)

    px = int(state.person_x * s)
    py = int(state.person_y * s)
    dx = int(state.drone_x  * s)
    dy = int(state.drone_y  * s)

    cv2.line(img, (dx, dy), (px, py), CLR_LINE, 2)
    cv2.ellipse(img, (px, py+20), (16,5), 0, 0, 360, (210,210,210), -1)
    cv2.circle(img, (px, py), 16, CLR_PERSON, -1)
    cv2.circle(img, (px, py), 16, (80,50,10), 2)
    cv2.putText(img, "P", (px-6, py+5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,255,255), 2, cv2.LINE_AA)

    for ang in [45, 135, 225, 315]:
        rad = np.radians(ang)
        ex  = int(dx + 14*np.cos(rad))
        ey  = int(dy + 14*np.sin(rad))
        cv2.line(img, (dx,dy), (ex,ey), (130,130,130), 2)
        cv2.circle(img, (ex,ey), 5, (160,160,160), -1)
    cv2.circle(img, (dx,dy), 9, CLR_DRONE, -1)
    cv2.circle(img, (dx,dy), 9, (10,10,120), 2)
    cv2.putText(img, "D", (dx-5,dy+4), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255,255,255), 2, cv2.LINE_AA)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ════════════════════════════════════════════════
#  APP
# ════════════════════════════════════════════════
class DroneApp:
    def __init__(self, root):
        self.root      = root
        self.state     = AppState()
        self.drone     = DroneController(self.state)
        self.smooth_x  = 0.5
        self.smooth_y  = 0.5
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
        tk.Label(tb, text="Autonomous Drone Umbrella",
                 font=("Courier New", 16, "bold"),
                 fg="#00e676", bg="#0f0f0f").pack(side="left")
        self.mode_badge = tk.Label(tb, text="  SIMULATION  ",
                 font=("Courier New", 10),
                 fg="#ffb300", bg="#1a1a1a", padx=6, pady=2)
        self.mode_badge.pack(side="right")

        tk.Label(self.root,
                 text="Detection: Face Detection (Haar Cascade) — no colored object needed",
                 font=("Courier New", 9), fg="#444", bg="#0f0f0f").pack(anchor="w", padx=16, pady=(0,6))

        content = tk.Frame(self.root, bg="#0f0f0f")
        content.pack(padx=16)

        left = tk.Frame(content, bg="#0f0f0f")
        left.pack(side="left", padx=(0,12))
        tk.Label(left, text="WEBCAM FEED",
                 font=("Courier New", 9), fg="#555", bg="#0f0f0f").pack(anchor="w")
        self.cam_label = tk.Label(left, bg="#1a1a1a",
                                  width=FRAME_W, height=FRAME_H)
        self.cam_label.pack()

        right = tk.Frame(content, bg="#0f0f0f")
        right.pack(side="left")
        tk.Label(right, text="SIMULATION VIEW",
                 font=("Courier New", 9), fg="#555", bg="#0f0f0f").pack(anchor="w")
        self.sim_label = tk.Label(right, bg="#f8f8f8",
                                  width=SIM_SIZE, height=SIM_SIZE)
        self.sim_label.pack()

        sf = tk.Frame(self.root, bg="#141414")
        sf.pack(fill="x", padx=16, pady=8)
        self.lbl_det  = self._stat(sf, "Detection",  "NO",           "#ef5350")
        self.lbl_trk  = self._stat(sf, "Tracking",   "PAUSED",       "#ffb300")
        self.lbl_pos  = self._stat(sf, "Person pos", "(0.50, 0.50)", "#aaa")
        self.lbl_drn  = self._stat(sf, "Drone pos",  "(0.50, 0.18)", "#aaa")
        self.lbl_err  = self._stat(sf, "Error X",    "0.00",         "#aaa")
        self.lbl_bat  = self._stat(sf, "Battery",    "0%",           "#aaa")
        self.lbl_fps  = self._stat(sf, "FPS",        "0",            "#aaa")
        self.lbl_stat = self._stat(sf, "Status",     "Ready",        "#00e676")

        bf = tk.Frame(self.root, bg="#0f0f0f")
        bf.pack(padx=16, pady=(0,10), fill="x")
        self.btn_start   = self._btn(bf, "START TRACKING", "#00e676", "#000", self._start)
        self.btn_pause   = self._btn(bf, "PAUSE",          "#ffb300", "#000", self._pause)
        self.btn_reset   = self._btn(bf, "RESET",          "#2196f3", "#fff", self._reset)
        self.btn_takeoff = self._btn(bf, "TAKEOFF",        "#7c4dff", "#fff", self._takeoff_land)
        self.btn_drone   = self._btn(bf, "CONNECT DRONE",  "#1a1a1a", "#00e676", self._toggle_drone)
        self.btn_quit    = self._btn(bf, "QUIT",           "#ef5350", "#fff", self._quit)
        for b in [self.btn_start, self.btn_pause, self.btn_reset,
                  self.btn_takeoff, self.btn_drone, self.btn_quit]:
            b.pack(side="left", padx=4, pady=4)

        sf2 = tk.Frame(self.root, bg="#0f0f0f")
        sf2.pack(padx=16, pady=(0,16), fill="x")
        tk.Label(sf2, text="Tracking sensitivity",
                 font=("Courier New", 9), fg="#555", bg="#0f0f0f").pack(side="left", padx=(0,10))
        self.sens_var = tk.DoubleVar(value=0.3)
        tk.Scale(sf2, from_=0.05, to=1.0, resolution=0.05,
                 variable=self.sens_var, orient="horizontal", length=200,
                 bg="#0f0f0f", fg="#aaa", troughcolor="#1a1a1a",
                 highlightthickness=0, bd=0,
                 command=lambda v: self.state.set(sensitivity=float(v))).pack(side="left")
        self.sens_lbl = tk.Label(sf2, text="0.30",
                                 font=("Courier New", 9), fg="#aaa", bg="#0f0f0f")
        self.sens_lbl.pack(side="left", padx=8)

    def _stat(self, p, key, val, color):
        f = tk.Frame(p, bg="#141414", padx=12, pady=8)
        f.pack(side="left", expand=True, fill="x", padx=4, pady=6)
        tk.Label(f, text=key.upper(), font=("Courier New", 8),
                 fg="#444", bg="#141414").pack(anchor="w")
        lbl = tk.Label(f, text=val, font=("Courier New", 11, "bold"),
                       fg=color, bg="#141414")
        lbl.pack(anchor="w")
        return lbl

    def _btn(self, p, text, bg, fg, cmd):
        return tk.Button(p, text=text, font=("Courier New", 9, "bold"),
                         bg=bg, fg=fg, relief="flat", padx=12, pady=6,
                         cursor="hand2", activebackground=bg, command=cmd)

    def _start(self):
        self.state.set(tracking=True, status="Tracking")
        self.lbl_trk.config(text="ACTIVE", fg="#00e676")

    def _pause(self):
        self.state.set(tracking=False, status="Paused")
        self.lbl_trk.config(text="PAUSED", fg="#ffb300")
        if self.drone.active and self.state.airborne:
            self.drone.send_rc(0)

    def _reset(self):
        self.state.set(tracking=False, person_x=0.5, person_y=0.5,
                       drone_x=0.5, drone_y=0.18, status="Reset")
        self.smooth_x = 0.5
        self.smooth_y = 0.5
        self.lbl_trk.config(text="PAUSED", fg="#ffb300")

    def _takeoff_land(self):
        if not self.state.drone_mode:
            self.lbl_stat.config(text="Connect drone first!", fg="#ef5350")
            return
        if not self.state.airborne:
            self.drone.takeoff()
            self.btn_takeoff.config(text="LAND", bg="#ef5350")
        else:
            self.drone.land()
            self.btn_takeoff.config(text="TAKEOFF", bg="#7c4dff")

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
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.fps_timer = time.time()
        self.fps_count = 0
        threading.Thread(target=self._cam_loop, daemon=True).start()

    def _cam_loop(self):
        while self.state.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]

            cx, cy, bbox = detect_face(frame)
            detected = cx is not None

            if detected:
                nx   = cx / fw
                ny   = cy / fh
                sens = self.state.sensitivity
                self.smooth_x = self.smooth_x*(1-sens) + nx*sens
                self.smooth_y = self.smooth_y*(1-sens) + ny*sens
                error_x = (self.smooth_x - 0.5) * 2.0
                self.state.set(person_x=self.smooth_x, person_y=self.smooth_y,
                               detected=True, error_x=error_x, status="Tracking")
                if bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x,y), (x+w,y+h), CLR_BOX, 2)
                    L = 15
                    for (x1,y1,x2,y2) in [
                        (x,y,x+L,y),(x,y,x,y+L),
                        (x+w,y,x+w-L,y),(x+w,y,x+w,y+L),
                        (x,y+h,x+L,y+h),(x,y+h,x,y+h-L),
                        (x+w,y+h,x+w-L,y+h),(x+w,y+h,x+w,y+h-L)]:
                        cv2.line(frame, (x1,y1), (x2,y2), CLR_BOX, 3)
                    cv2.circle(frame, (cx,cy), 4, CLR_BOX, -1)
                    cv2.putText(frame, f"FACE ({cx},{cy})", (x, y-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_BOX, 1, cv2.LINE_AA)
                if self.state.airborne:
                    if self.state.tracking and detected:
                        spd = int(np.clip(error_x * 30, -30, 30))
                        self.drone.send_rc(-spd)
                    else:
                        self.drone.send_rc(0)
                time.sleep(0.1)

            self.state.drone_x = self.state.drone_x*0.88 + self.state.person_x*0.12
            self.state.drone_y = 0.18

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
            img   = Image.fromarray(self.cam_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imgtk
            self.cam_label.config(image=imgtk)

        sim_arr = make_sim(s)
        sim_img = Image.fromarray(sim_arr)
        sim_tk  = ImageTk.PhotoImage(image=sim_img)
        self.sim_label.imgtk = sim_tk
        self.sim_label.config(image=sim_tk)

        self.lbl_det.config(text="YES" if s.detected else "NO",
                            fg="#00e676" if s.detected else "#ef5350")
        self.lbl_pos.config(text=f"({s.person_x:.2f}, {s.person_y:.2f})")
        self.lbl_drn.config(text=f"({s.drone_x:.2f}, {s.drone_y:.2f})")
        self.lbl_err.config(text=f"{s.error_x:+.2f}")
        self.lbl_bat.config(text=f"{s.battery}%")
        self.lbl_fps.config(text=str(s.fps))
        self.lbl_stat.config(text=s.status)
        self.sens_lbl.config(text=f"{self.sens_var.get():.2f}")

        self.root.after(33, self._update_loop)


# ════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = DroneApp(root)
    root.protocol("WM_DELETE_WINDOW", app._quit)
    root.mainloop()
    if app.cap.isOpened():
        app.cap.release()
    print("Goodbye!")