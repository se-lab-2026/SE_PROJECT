# ============================================================
# renderer.py — Everything that draws pixels on screen
#
# Two main outputs:
#   1. The webcam feed with a status overlay (face box, HUD)
#   2. The top-down simulation view showing drone & person positions
#
# All drawing helpers are here. Nothing in this file sends
# commands to the drone or modifies state — it's purely visual.
# ============================================================

import cv2
import numpy as np

from state import AppState
from config import (
    SIM_SIZE,
    CLR_DRONE, CLR_PERSON, CLR_LINE, CLR_BOX,
    CLR_TEXT, CLR_PANEL, CLR_GRID,
    CLR_TRACKING, CLR_LOST, CLR_WARN
)


# ── Low-level Drawing Primitives ─────────────────────────────

def draw_panel(img, x, y, w, h, alpha=0.6):
    """
    Draw a semi-transparent dark panel — used as a background
    behind text so it's readable over any camera content.
    """
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), CLR_PANEL, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)


def put_text(img, text, pos, color=CLR_TEXT, scale=0.55, thickness=1):
    """
    Shorthand for cv2.putText with our default font.
    Keeps the rest of the code from repeating FONT_HERSHEY_SIMPLEX everywhere.
    """
    cv2.putText(
        img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA
    )


# ── Webcam Feed Overlay ──────────────────────────────────────

def draw_status_overlay(frame, state: AppState):
    """
    Draws the heads-up display on top of the live webcam frame:
      - Top bar: app name, FPS, tracking state, mode, battery
      - Bottom bar: keyboard shortcut reminder
      - Middle area (when face found): error values and face area
    """
    h, w = frame.shape[:2]

    # Top status bar
    draw_panel(frame, 0, 0, w, 75)

    tracking_color = CLR_TRACKING if state.detected else CLR_LOST
    tracking_label = "TRACKING" if state.detected else "LOST"

    put_text(frame, "Autonomous Drone Umbrella",   (10, 20),  CLR_TEXT, 0.6, 1)
    put_text(frame, f"FPS: {state.fps}",           (10, 45),  CLR_TEXT, 0.45)
    put_text(frame, f"State: {tracking_label}",   (110, 45), tracking_color, 0.45)
    put_text(frame, f"Mode: {'DRONE' if state.drone_mode else 'SIM'}", (230, 45),
             CLR_WARN if state.drone_mode else CLR_TEXT, 0.45)
    put_text(frame, f"Battery: {state.battery}%", (340, 45), CLR_TEXT, 0.45)

    # Big "AIRBORNE" warning when we're actually flying
    if state.airborne:
        put_text(frame, "AIRBORNE", (530, 20), CLR_WARN, 0.55, 2)

    # Bottom shortcut bar — always visible so user doesn't forget controls
    draw_panel(frame, 0, h - 40, w, 40)
    put_text(frame, "SPACE:Takeoff/Land  S:Track  P:Pause  R:Reset  D:Drone Mode  Q:Quit",
             (8, h - 15), CLR_TEXT, 0.4)

    # Error readout row — only meaningful when we have a face to track
    if state.detected:
        put_text(frame, f"X Error: {state.error_x:+.2f}", (10,  h - 55), CLR_TRACKING, 0.45)
        put_text(frame, f"Z Error: {state.error_z:+.2f}", (150, h - 55), CLR_WARN,     0.45)
        put_text(frame, f"Area: {state.area}",            (290, h - 55), CLR_TEXT,     0.45)
        put_text(frame, f"Depth: {state.depth_state}",    (400, h - 55), CLR_TEXT,     0.45)


# ── Top-down Simulation View ─────────────────────────────────

def draw_simulation(state: AppState):
    """
    Renders the bird's-eye simulation window showing where the
    drone and person are relative to each other.

    The sim coordinates are normalized (0–1) and scaled to SIM_SIZE pixels.
    Person is the orange circle (P), drone is the blue circle (D).
    The line between them shows the tracking vector.
    """
    # Off-white background so it doesn't look like a debug console
    sim = np.ones((SIM_SIZE, SIM_SIZE, 3), dtype=np.uint8) * 245
    sim[:] = (245, 245, 242)

    # Grid lines — helps judge relative position at a glance
    step = SIM_SIZE // 10
    for i in range(0, SIM_SIZE, step):
        cv2.line(sim, (i, 0),       (i, SIM_SIZE), CLR_GRID, 1)
        cv2.line(sim, (0, i),       (SIM_SIZE, i), CLR_GRID, 1)

    # Brighter center crosshair
    cx = SIM_SIZE // 2
    cv2.line(sim, (cx, 0),         (cx, SIM_SIZE), (180, 180, 180), 1)
    cv2.line(sim, (0,  cx),        (SIM_SIZE, cx), (180, 180, 180), 1)

    # Convert normalized positions to pixel coordinates
    px = int(state.person_x * SIM_SIZE)
    py = int(state.person_y * SIM_SIZE)
    dx = int(state.drone_x  * SIM_SIZE)
    dy = int(state.drone_y  * SIM_SIZE)

    # Line from drone to person — shows the tracking vector
    cv2.line(sim, (dx, dy), (px, py), CLR_LINE, 2)

    # Person: shadow ellipse first, then the circle on top
    cv2.ellipse(sim, (px, py + 22), (18, 6), 0, 0, 360, (200, 200, 200), -1)
    cv2.circle(sim,  (px, py), 18, CLR_PERSON, -1)
    cv2.circle(sim,  (px, py), 18, (80, 60, 20), 2)
    put_text(sim, "P", (px - 5, py + 5), (255, 255, 255), 0.55, 2)

    # Drone: four propeller arms at 45° angles, then body circle on top
    for angle in [45, 135, 225, 315]:
        rad = np.radians(angle)
        ex = int(dx + 16 * np.cos(rad))
        ey = int(dy + 16 * np.sin(rad))
        cv2.line(sim, (dx, dy), (ex, ey), (100, 100, 100), 2)
        cv2.circle(sim, (ex, ey), 5, (140, 140, 140), -1)

    cv2.circle(sim, (dx, dy), 10, CLR_DRONE, -1)
    cv2.circle(sim, (dx, dy), 10, (10, 10, 120), 2)
    put_text(sim, "D", (dx - 5, dy + 5), (255, 255, 255), 0.45, 2)

    # Bottom info panel — live telemetry in the sim window
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

    # Top title bar
    draw_panel(sim, 0, 0, SIM_SIZE, 30, alpha=0.7)
    put_text(sim, "Drone Umbrella — Top-down simulation", (10, 20), CLR_TEXT, 0.48)

    return sim
