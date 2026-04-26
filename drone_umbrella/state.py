# ============================================================
# state.py — The single source of truth for the whole app
#
# Instead of scattering variables everywhere, everything the
# system cares about lives here. The .set() method uses a lock
# so the RC thread and the main loop can both update state
# without stepping on each other.
# ============================================================

import threading


class AppState:
    def __init__(self):
        # ── Control Flags ────────────────────────────────────
        self.tracking   = False   # Is the drone actively chasing the person?
        self.airborne   = False   # Has it taken off?
        self.drone_mode = False   # Real Tello connected, or just simulation?
        self.running    = True    # Main loop keeps going while this is True

        # ── Positions (normalized 0.0–1.0 within the frame) ──
        self.person_x = 0.5   # Person starts at center of screen
        self.person_y = 0.5
        self.drone_x  = 0.5   # Drone starts just above center in sim view
        self.drone_y  = 0.2

        # ── Live Telemetry ───────────────────────────────────
        self.battery     = 0         # Tello battery % (0 in sim mode)
        self.detected    = False     # Did we find a face this frame?
        self.fps         = 0         # Frames per second of the main loop
        self.error_x     = 0.0       # Horizontal offset from center (-1 to +1)
        self.error_z     = 0.0       # Depth error — positive = too far away
        self.area        = 0         # Face bounding box area in pixels
        self.depth_state = "CENTERED"  # Human-readable depth status
        self.status_msg  = "Ready"   # One-liner shown in the UI

        # ── Thread Safety ────────────────────────────────────
        # The RC thread reads state while the main loop writes it.
        # This lock prevents torn reads (e.g. half-updated position).
        self._lock = threading.Lock()

    def set(self, **kwargs):
        """
        Update one or more state fields safely from any thread.
        Usage: state.set(tracking=True, status_msg="Tracking")
        """
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
