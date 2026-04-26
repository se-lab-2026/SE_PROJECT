# Drone Umbrella — Project Guide

## Project Structure

```
drone_umbrella/
│
├── main.py         ← START HERE — main loop + keyboard controls
├── config.py       ← All settings in one place (speeds, camera, colors)
├── state.py        ← Shared app state (thread-safe)
├── drone.py        ← Tello connection, takeoff, RC commands
├── detection.py    ← Face detection (OpenCV Haar cascade)
├── control.py      ← Speed math, deadzone logic
└── renderer.py     ← All drawing (webcam HUD + simulation view)
```

---

## How to Run

### 1. Install dependencies

```bash
pip install opencv-python numpy
pip install djitellopy        # Only needed for real drone — skip for sim mode
```

### 2. Run the app

```bash
cd drone_umbrella
python main.py
```

---

## Controls

| Key   | Action                              |
|-------|-------------------------------------|
| D     | Toggle drone mode / connect Tello   |
| SPACE | Takeoff (or Land if airborne)       |
| S     | Start / resume tracking             |
| P     | Pause tracking (drone hovers)       |
| R     | Reset simulation positions          |
| Q/ESC | Emergency stop and quit             |

---

## Running Without a Drone (Simulation Mode)

Just run `python main.py` without pressing D.  
The simulation window shows how the drone *would* move based on your face position.  
No Tello needed — great for testing the tracking logic.

---

## Tuning Tips

All tunable values are in **`config.py`**:

| Setting        | What it does                                         |
|----------------|------------------------------------------------------|
| `WEBCAM_INDEX` | Change to `1` if your webcam doesn't open            |
| `TARGET_AREA`  | Face pixel area at ideal hover distance — tune this  |
| `LR_INVERT`    | Flip to `False` if drone goes the wrong direction    |
| `FB_DEADZONE`  | Bigger = less forward/back jitter                    |
| `SPEED_LR/FB`  | How fast the drone chases the target                 |

---

## Real Drone Checklist

1. Charge Tello battery (check LED — solid white = ready)
2. Connect your PC to the Tello's WiFi (`TELLO-XXXXXX`)
3. Run `python main.py`
4. Press **D** to connect → watch terminal for `[DRONE] Connected! Battery: XX%`
5. Press **SPACE** to take off — tracking auto-starts
6. Press **SPACE** again to land, or **Q** to emergency stop
