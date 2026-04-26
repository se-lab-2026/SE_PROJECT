# ============================================================
# control.py — Speed math and deadzone helpers
#
# These small functions sit between "we have an error value"
# and "we tell the drone how fast to go". The deadzone logic
# is what keeps the drone from jittering when it's nearly
# centered — small errors get swallowed, only real offsets
# produce movement commands.
# ============================================================

import numpy as np
from config import (
    SPEED_LR, SPEED_FB,
    FB_DEADZONE, FB_MIN_SPEED, LR_DEADZONE
)


def apply_deadzone(value, deadzone):
    """
    If the error is tiny (within the deadzone band), treat it
    as zero — the drone stays put rather than micro-jittering.
    Once outside the band, we return the raw value unchanged.
    """
    if abs(value) < deadzone:
        return 0.0
    return value


def scaled_lr_speed(error_x: float) -> int:
    """
    Convert a normalized horizontal error (-1.0 to +1.0) into
    a left/right RC command (-SPEED_LR to +SPEED_LR).

    Positive result  → drone moves right
    Negative result  → drone moves left
    Zero             → person is close enough to center, don't bother
    """
    ex = apply_deadzone(error_x, LR_DEADZONE)
    if ex == 0:
        return 0
    return int(np.clip(ex * SPEED_LR, -SPEED_LR, SPEED_LR))


def scaled_fb_speed(error_z: float) -> int:
    """
    Convert a depth error (-1.5 to +1.5) into a forward/back
    RC command.

    Positive error → person is too far  → drone moves forward
    Negative error → person too close   → drone backs up

    The FB_MIN_SPEED floor ensures the drone actually moves
    instead of sending a signal too weak for the motors to act on.
    """
    ez = apply_deadzone(error_z, FB_DEADZONE)
    if ez == 0:
        return 0

    raw = int(np.clip(ez * SPEED_FB, -SPEED_FB, SPEED_FB))

    # Enforce minimum motor speed — below this the drone barely budges
    if raw > 0:
        return max(raw, FB_MIN_SPEED)
    return min(raw, -FB_MIN_SPEED)
