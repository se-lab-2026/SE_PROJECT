# ============================================================
# main.py — Entry point. Run this file to start the system.
#
# Controls:
#   D      → Toggle Drone Mode ON/OFF (connects to Tello)
#   SPACE  → Takeoff / Land
#   S      → Start tracking
#   P      → Pause tracking
#   R      → Reset simulation positions
#   Q/ESC  → Emergency stop and quit
# ============================================================

import cv2
import numpy as np
import time

from config import (
    WEBCAM_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    PERSON_SMOOTH, DRONE_SMOOTH,
    TARGET_AREA, FB_DEADZONE, LR_INVERT,
    CLR_BOX
)
from state import AppState
from drone import DroneController
from detection import detect_face
from control import scaled_lr_speed, scaled_fb_speed
from renderer import draw_status_overlay, draw_simulation, put_text


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

    # ── Initialization ───────────────────────────────────────
    state      = AppState()
    drone_ctrl = DroneController(state)

    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam index {WEBCAM_INDEX}. Try 1 instead.")
        return

    print("[OK] Webcam opened.")

    # Set up both windows side by side
    cv2.namedWindow("Webcam Feed",     cv2.WINDOW_NORMAL)
    cv2.namedWindow("Simulation View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam Feed",     FRAME_WIDTH, FRAME_HEIGHT)
    cv2.resizeWindow("Simulation View", 500, 500)
    cv2.moveWindow("Webcam Feed",      0, 0)
    cv2.moveWindow("Simulation View",  FRAME_WIDTH + 20, 0)

    # ── Local loop variables ─────────────────────────────────
    fps_timer = time.time()
    fps_count = 0
    smooth_x  = 0.5   # Smoothed face X position (exponential moving average)
    smooth_y  = 0.5
    error_x   = 0.0
    error_z   = 0.0

    # ── Main Loop ────────────────────────────────────────────
    while state.running:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read webcam frame.")
            break

        # Flip so it mirrors naturally — feels more intuitive for the operator
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        # ── Face Detection ───────────────────────────────────
        cx, cy, area, bbox = detect_face(frame)
        detected = cx is not None

        if detected:
            # Normalize to 0–1 range so math is frame-size independent
            nx = cx / fw
            ny = cy / fh

            # Exponential smoothing — damps jitter from the detector
            smooth_x = smooth_x * (1 - PERSON_SMOOTH) + nx * PERSON_SMOOTH
            smooth_y = smooth_y * (1 - PERSON_SMOOTH) + ny * PERSON_SMOOTH

            # Horizontal error: positive = person is right of center
            # LR_INVERT flips this because the laptop cam mirrors reality
            raw_error_x = (smooth_x - 0.5) * 2.0
            error_x     = -raw_error_x if LR_INVERT else raw_error_x

            # Depth error: positive = person is too far → move forward
            error_z = (TARGET_AREA - area) / TARGET_AREA
            error_z = float(np.clip(error_z, -1.5, 1.5))

            # Human-readable depth status for the HUD
            if   error_z >  FB_DEADZONE:  depth_state = "TOO FAR"
            elif error_z < -FB_DEADZONE:  depth_state = "TOO CLOSE"
            else:                          depth_state = "CENTERED"

            state.set(
                person_x    = smooth_x,
                person_y    = smooth_y,
                detected    = True,
                error_x     = error_x,
                error_z     = error_z,
                area        = area,
                depth_state = depth_state,
                status_msg  = "Tracking"
            )

            # Draw the face bounding box and center dot on the camera feed
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), CLR_BOX, 2)
                cv2.circle(frame, (cx, cy), 6, CLR_BOX, -1)
                put_text(frame, f"Face Area: {area}", (x, max(y - 8, 20)), CLR_BOX, 0.45)

            # ── Drone RC Commands ────────────────────────────
            if state.airborne:
                if state.tracking:
                    # Active tracking: compute and send movement command
                    lr = scaled_lr_speed(error_x)
                    fb = scaled_fb_speed(error_z)
                    print(f"[RC] lr={lr:+d} fb={fb:+d}  ex={error_x:.3f} ez={error_z:.3f}")
                    drone_ctrl.send_tracking_command(error_x, error_z)
                else:
                    # Paused: send zeros so Tello doesn't auto-land after 1s
                    drone_ctrl.stop_motion()

        else:
            # No face found — zero out errors so drone holds still
            error_x = 0.0
            error_z = 0.0
            state.set(
                detected    = False,
                error_x     = 0.0,
                error_z     = 0.0,
                area        = 0,
                depth_state = "NO FACE",
                status_msg  = "Searching..." if state.tracking else "Paused"
            )
            # Keep the Tello happy with a keep-alive even when nothing's detected
            if state.airborne:
                drone_ctrl.stop_motion()

        # ── Simulation Position Update ───────────────────────
        # In the top-down view, the drone icon smoothly follows the person
        state.drone_x = state.drone_x * (1 - DRONE_SMOOTH) + state.person_x * DRONE_SMOOTH

        # Drone sits "above" the person in the top-down view.
        # error_z nudges it closer/further to reflect depth corrections.
        target_drone_y = state.person_y - 0.15 - (state.error_z * 0.10)
        state.drone_y  = float(np.clip(
            state.drone_y * (1 - DRONE_SMOOTH) + target_drone_y * DRONE_SMOOTH,
            0.05, 0.95
        ))

        # ── FPS Counter ──────────────────────────────────────
        fps_count += 1
        if time.time() - fps_timer >= 1.0:
            state.set(fps=fps_count)
            fps_count = 0
            fps_timer = time.time()

        # ── Render ───────────────────────────────────────────
        draw_status_overlay(frame, state)
        sim = draw_simulation(state)

        cv2.imshow("Webcam Feed",     frame)
        cv2.imshow("Simulation View", sim)

        # ── Keyboard Input ───────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:   # Q or ESC
            print("[INFO] Quitting...")
            drone_ctrl.emergency_stop()
            break

        elif key == ord(' '):              # SPACE — takeoff or land
            if not state.airborne:
                # If drone mode is on but not yet connected, try connecting now
                if state.drone_mode and not drone_ctrl.active:
                    success = drone_ctrl.connect()
                    if not success:
                        print("[INFO] Could not connect. Staying in simulation mode.")
                        state.set(drone_mode=False)
                drone_ctrl.takeoff()
                # Auto-start tracking so the drone moves the moment it's up
                state.set(tracking=True, status_msg="Tracking")
                print("[INFO] Tracking auto-started after takeoff")
            else:
                state.set(tracking=False)
                drone_ctrl.land()

        elif key == ord('s'):              # S — start/resume tracking
            state.set(tracking=True, status_msg="Tracking")
            print("[INFO] Tracking started")

        elif key == ord('p'):              # P — pause, hold position
            state.set(tracking=False, status_msg="Paused")
            drone_ctrl.stop_motion()
            print("[INFO] Tracking paused")

        elif key == ord('r'):              # R — reset sim positions
            state.set(
                tracking    = False,
                person_x    = 0.5,
                person_y    = 0.5,
                drone_x     = 0.5,
                drone_y     = 0.18,
                error_x     = 0.0,
                error_z     = 0.0,
                area        = 0,
                depth_state = "CENTERED",
                status_msg  = "Reset"
            )
            smooth_x = 0.5
            smooth_y = 0.5
            drone_ctrl.stop_motion()
            print("[INFO] Reset")

        elif key == ord('d'):              # D — toggle drone mode
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

    # ── Cleanup ──────────────────────────────────────────────
    print("[INFO] Cleaning up...")
    drone_ctrl.disconnect()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done. Goodbye!")


if __name__ == "__main__":
    main()
