# ============================================================
# drone.py — Everything that touches the actual Tello drone
#
# This class is intentionally the only place in the codebase
# that talks to djitellopy. If you're in simulation mode, all
# the "real" calls are skipped and state updates still happen
# so the rest of the UI behaves normally.
#
# The RC loop runs in its own thread at 20 Hz — that's the
# Tello's reliable command rate. The main loop just updates
# the desired speed values; the thread delivers them.
# ============================================================

import time
import threading

from state import AppState
from control import scaled_lr_speed, scaled_fb_speed
from config import RC_INTERVAL, SPEED_UD, SPEED_YAW

# Try to import djitellopy — if it's missing we stay in sim mode gracefully
try:
    from djitellopy import Tello
    TELLO_AVAILABLE = True
except ImportError:
    TELLO_AVAILABLE = False
    print("[WARNING] djitellopy not found. Running in simulation-only mode.")
    print("[INFO] Install with: pip install djitellopy")


class DroneController:
    def __init__(self, state: AppState):
        self.state     = state
        self.tello     = None
        self.active    = False   # True only when a real Tello is connected

        # These four values are the "desired" RC state.
        # The RC thread reads them continuously and fires them to Tello.
        self._lr  = 0
        self._fb  = 0
        self._ud  = 0
        self._yaw = 0

        self._rc_lock   = threading.Lock()
        self._rc_thread = None

    # ── RC Thread ────────────────────────────────────────────

    def _rc_loop(self):
        """
        Runs at 20 Hz while the drone is airborne.
        Keeps feeding the Tello fresh RC values so it doesn't
        auto-land after ~1 second of silence.
        """
        while self.active and self.state.airborne:
            with self._rc_lock:
                lr, fb, ud, yaw = self._lr, self._fb, self._ud, self._yaw
            try:
                self.tello.send_rc_control(lr, fb, ud, yaw)
            except Exception as e:
                print(f"[DRONE] RC error: {e}")
            time.sleep(RC_INTERVAL)

    def set_rc(self, lr, fb, ud=0, yaw=0):
        """
        Update the desired RC values. Safe to call from any thread.
        The RC loop will pick these up on its next tick (~50ms later).
        """
        with self._rc_lock:
            self._lr  = int(lr)
            self._fb  = int(fb)
            self._ud  = int(ud)
            self._yaw = int(yaw)

    # ── Connection ───────────────────────────────────────────

    def connect(self):
        """
        Attempt to connect to the Tello over WiFi.
        Returns True on success, False if anything goes wrong.
        Make sure you're connected to the Tello's WiFi before calling this.
        """
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

    # ── Flight ───────────────────────────────────────────────

    def takeoff(self):
        """
        Take off if a real drone is connected, or fake it in sim mode.
        After takeoff we spin up the RC thread so the Tello keeps
        receiving keep-alive commands at 20 Hz.
        """
        if self.active and self.tello:
            try:
                print("[DRONE] Taking off...")
                self.tello.takeoff()
                time.sleep(1)   # Give it a moment to stabilize
                self.state.set(airborne=True, status_msg="Airborne")
                try:
                    self.state.set(battery=self.tello.get_battery())
                except Exception:
                    pass
                # Start firing RC commands now that we're in the air
                self._rc_thread = threading.Thread(target=self._rc_loop, daemon=True)
                self._rc_thread.start()
            except Exception as e:
                print(f"[DRONE] Takeoff error: {e}")
        else:
            # Simulation: just flip the flag, nothing physically happens
            self.state.set(airborne=True, status_msg="Airborne (sim)")

    def land(self):
        """
        Stop motion and land. In sim mode, just marks us as grounded.
        """
        if self.active and self.tello:
            try:
                print("[DRONE] Landing...")
                self.set_rc(0, 0, 0, 0)   # Stop first, then land cleanly
                time.sleep(0.1)
                self.tello.land()
                self.state.set(airborne=False, status_msg="Landed")
            except Exception as e:
                print(f"[DRONE] Land error: {e}")
        else:
            self.state.set(airborne=False, status_msg="Landed (sim)")

    def emergency_stop(self):
        """
        Best-effort emergency: zero RC, attempt to land, then mark stopped.
        Called on Q / ESC — we try our best even if things are broken.
        """
        try:
            if self.active and self.tello:
                self.set_rc(0, 0, 0, 0)
                time.sleep(0.1)
                if self.state.airborne:
                    self.tello.land()
        except Exception:
            pass   # If this fails, we're quitting anyway
        self.state.set(airborne=False, running=False)

    # ── Tracking Commands ────────────────────────────────────

    def send_tracking_command(self, error_x: float, error_z: float):
        """
        Convert face-tracking errors into RC speeds and queue them.
        Called every frame when tracking is active and drone is airborne.
        """
        lr = scaled_lr_speed(error_x)
        fb = scaled_fb_speed(error_z)
        self.set_rc(lr, fb, SPEED_UD, SPEED_YAW)

    def stop_motion(self):
        """
        Send zeros — drone hovers in place.
        Used when tracking is paused or face is lost.
        (Also acts as a keep-alive so Tello doesn't auto-land.)
        """
        self.set_rc(0, 0, 0, 0)

    # ── Cleanup ──────────────────────────────────────────────

    def disconnect(self):
        """
        Gracefully land (if still airborne) and close the Tello connection.
        Called on normal exit.
        """
        if self.active and self.tello:
            try:
                self.set_rc(0, 0, 0, 0)
                time.sleep(0.1)
                if self.state.airborne:
                    self.tello.land()
                self.tello.end()
            except Exception:
                pass
