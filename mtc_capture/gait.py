"""Procedural footsteps for viewing takes without leg tracking.

The base path (from the operator's head) is known for the whole take, so steps are planned
ahead: a foot steps when its planted pose is too far from where it should be under the body
a little later (offset rotated with the base yaw). One foot swings at a time, alternating
when both need to move, with a short double support between steps; planted feet do not slide.
Steps get quicker when the body moves fast (cadence ~ speed). Swing: smoothstep in x, y and yaw,
sine lift. This is a visual gait, not a balance model.
"""

import numpy as np

SIDES = ("left", "right")


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def plan_footsteps(t, base_xy, base_yaw, offsets, step_time=0.38, min_step_time=0.24, lift=0.07, trigger=0.035,
                   trigger_yaw=0.15, max_step=0.45, double_support=0.05, lookahead=1.8):
    """t (n,), base_xy (n,2), base_yaw (n,), offsets {side: (2,) foot xy in the base frame}
    -> feet (n, 2, 4): x, y, lift z, yaw per side; swing (n, 2) bool."""
    n = len(t)
    R = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def nominal(i, k):
        return base_xy[i] + R(base_yaw[i]) @ offsets[SIDES[k]]

    planted = [np.r_[nominal(0, k), base_yaw[0]] for k in range(2)]
    feet, swinging = np.zeros((n, 2, 4)), np.zeros((n, 2), bool)
    swing, last_land, last_side = None, -np.inf, None
    for i in range(n):
        if swing is not None and t[i] >= swing["t0"] + swing["T"]:
            planted[swing["k"]] = swing["to"]
            last_land, last_side, swing = swing["t0"] + swing["T"], swing["k"], None
        if swing is None and t[i] - last_land >= double_support:
            j0 = min(int(np.searchsorted(t, t[i] + step_time)), n - 1)
            speed = np.linalg.norm(base_xy[j0] - base_xy[i]) / max(t[j0] - t[i], 1e-3)
            T = float(np.clip(0.16 / max(speed, 1e-3), min_step_time, step_time))
            j = min(int(np.searchsorted(t, t[i] + lookahead * T)), n - 1)
            dev = []
            for k in range(2):
                d = np.linalg.norm(nominal(j, k) - planted[k][:2])
                dy = abs(_wrap(base_yaw[j] - planted[k][2]))
                dev.append(max(d / trigger, dy / trigger_yaw))
            need = [k for k in range(2) if dev[k] > 1]
            if need:
                k = next((k for k in need if k != last_side), max(need, key=lambda k: dev[k]))
                goal = nominal(j, k)
                step = goal - planted[k][:2]
                if np.linalg.norm(step) > max_step:
                    goal = planted[k][:2] + step * max_step / np.linalg.norm(step)
                yaw = planted[k][2] + _wrap(base_yaw[j] - planted[k][2])
                swing = dict(k=k, t0=t[i], T=T, frm=planted[k].copy(), to=np.r_[goal, yaw])
        for k in range(2):
            feet[i, k] = np.r_[planted[k][:2], 0.0, planted[k][2]]
        if swing is not None:
            s = np.clip((t[i] - swing["t0"]) / swing["T"], 0, 1)
            e = s * s * (3 - 2 * s)
            a, b = swing["frm"], swing["to"]
            feet[i, swing["k"]] = np.r_[a[:2] + e * (b[:2] - a[:2]), lift * np.sin(np.pi * s), a[2] + e * (b[2] - a[2])]
            swinging[i, swing["k"]] = True
    return feet, swinging
