"""Frames, per-segment anchoring and live clearance of a G1-sized proxy from raw XR tracking.

Used for operator feedback and per-sample safety flags; the recorded motion stays raw.

Frames:
    XR     WebXR local-floor world (OpenXR basis: y up, -z forward), metres, human scale.
    home   the operator's calibrated spot: origin on the floor under the head, +x along
           the gaze at calibration, z up, G1 scale.
    scene  the furniture scene.json frame: metres, z up, origin at the room corner, G1 scale.
Each segment maps its start pose (x, y, yaw) onto home, so
    T_xr_from_scene = T_xr_from_home @ T_home_from_scene(segment start)    (similarity, scale 1/alpha)

Proxy (G1 scale):
    hand     the Dex3 collision envelope of this branch: a capsule along the tracked hand's
             axis (OpenXR wrist -Z, towards the fingers) from 0.037 to 0.222 m past the wrist
             with the envelope's 5.2 cm half-thickness. The robot's fingers are fixed, so
             human finger motion does not change it.
    forearm  a 32 mm capsule 0.184 m from the wrist along +Z (towards the elbow), or towards the
             tracked elbow when the native app's body tracking provides one. Without it, a strongly
             bent wrist moves this capsule off the real forearm.
    body     head sphere + torso capsule under the head. Legs are not tracked.
"""

import numpy as np

from .g1_body import (DEX3_ENVELOPE_END, DEX3_ENVELOPE_RADIUS, DEX3_ENVELOPE_START, FOREARM_LENGTH, FOREARM_RADIUS,
                      G1_EYE_HEIGHT, G1_PELVIS_HEIGHT, G1_SHOULDER_HEIGHT, HEAD_RADIUS, TORSO_RADIUS)

# z-up axes expressed in the y-up XR basis: x -> x, y -> -z, z -> y
R_XR_FROM_ZUP = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


def home_transform(head_xr, alpha):
    """T_xr_from_home (4x4 similarity) and the XR yaw of the gaze at calibration."""
    fwd = -head_xr[:3, 2]                        # the camera looks along -z
    yaw = float(np.arctan2(-fwd[2], fwd[0]))     # rotation about XR +y taking +x onto the gaze
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ R_XR_FROM_ZUP / alpha
    T[:3, 3] = [head_xr[0, 3], 0.0, head_xr[2, 3]]  # the local-floor floor is y = 0
    return T, yaw


def home_from_scene(start):
    """T_home_from_scene: the segment start pose (x, y, yaw) becomes the home origin."""
    x, y, yaw = start
    c, s = np.cos(-yaw), np.sin(-yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = -T[:2, :2] @ [x, y]
    return T


def invert_similarity(T):
    A = T[:3, :3]
    k = np.cbrt(np.linalg.det(A))
    Ti = np.eye(4)
    Ti[:3, :3] = A.T / k ** 2
    Ti[:3, 3] = -Ti[:3, :3] @ T[:3, 3]
    return Ti


def to_scene(T_scene_from_xr, pts):
    pts = np.asarray(pts, float)
    return pts @ T_scene_from_xr[:3, :3].T + T_scene_from_xr[:3, 3]


def rot_to_scene(T_scene_from_xr, R):
    """Rotation part only (scale removed)."""
    A = T_scene_from_xr[:3, :3]
    return (A / np.cbrt(np.linalg.det(A))) @ R


class ObstacleSet:
    """Oriented boxes of a scene for vectorised signed-distance queries."""

    def __init__(self, scene):
        self.c, self.h, yaw, self.asset = scene.obstacle_arrays()
        cs, sn = np.cos(yaw), np.sin(yaw)
        self.Rt = np.zeros((len(yaw), 3, 3))        # world -> box local
        self.Rt[:, 0, 0], self.Rt[:, 0, 1] = cs, sn
        self.Rt[:, 1, 0], self.Rt[:, 1, 1] = -sn, cs
        self.Rt[:, 2, 2] = 1.0

    def signed_distance(self, pts):
        """(n,3) points -> (n, m) signed distance to each box (negative inside)."""
        d = pts[:, None, :] - self.c[None]
        local = np.einsum("mij,nmj->nmi", self.Rt, d)
        q = np.abs(local) - self.h[None]
        return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)

    def min_distance(self, pts, radii):
        """Minimum clearance of spheres (pts, radii) -> (distance, asset id or -1)."""
        pts = np.asarray(pts, float).reshape(-1, 3)
        if len(pts) == 0 or len(self.c) == 0 or not np.all(np.isfinite(pts)):
            return np.nan, -1
        d = self.signed_distance(pts) - np.asarray(radii, float).reshape(-1, 1)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        return float(d[i, j]), int(self.asset[j])


def _capsule(p0, direction, a, b, radius):
    """Spheres covering a capsule from p0 + a*dir to p0 + b*dir (surface to surface)."""
    if b - a <= 2 * radius:
        ts = np.array([(a + b) / 2])
    else:
        ts = np.linspace(a + radius, b - radius, max(2, int(np.ceil((b - a - 2 * radius) / radius)) + 1))
    return p0[None] + ts[:, None] * direction[None], np.full(len(ts), radius)


def hand_proxy(T_scene_from_xr, wrist_xr, elbow_xr=None):
    """Hand tracking (OpenXR wrist joint) -> dict(hand=(pts, r), forearm=(pts, r)) in the scene.
    With a tracked elbow (native app: PICO body tracking), the forearm points at it instead of along
    the wrist's +Z, so a bent wrist no longer moves the forearm capsule."""
    wrist = to_scene(T_scene_from_xr, wrist_xr[:3, 3])
    R = rot_to_scene(T_scene_from_xr, wrist_xr[:3, :3])
    fore = R[:, 2]
    if elbow_xr is not None and np.all(np.isfinite(elbow_xr)):
        d = to_scene(T_scene_from_xr, np.asarray(elbow_xr, float)) - wrist
        if np.linalg.norm(d) > 0.05:
            fore = d / np.linalg.norm(d)
    return {"hand": _capsule(wrist, -R[:, 2], DEX3_ENVELOPE_START, DEX3_ENVELOPE_END, DEX3_ENVELOPE_RADIUS),
            "forearm": _capsule(wrist, fore, 0.0, FOREARM_LENGTH, FOREARM_RADIUS)}


def controller_proxy(T_scene_from_xr, ctrl_xr):
    """Controller grip pose (Unitree arm convention, +x wrist -> fingers)."""
    wrist = to_scene(T_scene_from_xr, ctrl_xr[:3, 3])
    R = rot_to_scene(T_scene_from_xr, ctrl_xr[:3, :3])
    return {"hand": _capsule(wrist, R[:, 0], DEX3_ENVELOPE_START, DEX3_ENVELOPE_END, DEX3_ENVELOPE_RADIUS),
            "forearm": _capsule(wrist, -R[:, 0], 0.0, FOREARM_LENGTH, FOREARM_RADIUS)}


# G1 upper body (MuJoCo collision geoms of the branch model): torso capsule radius 0.115 (round),
# shoulder spheres 0.055 at +-0.10 m sideways, upper arms 0.05 thick and ~0.21 m long; so the robot
# is ~0.23 m deep but ~0.44 m wide across the shoulders and arms.
G1_TORSO_RADIUS = 0.115
G1_SHOULDER_OFFSET, G1_SHOULDER_RADIUS = 0.10, 0.055
G1_UPPER_ARM_LENGTH, G1_UPPER_ARM_RADIUS = 0.21, 0.05


def body_proxy(T_scene_from_xr, head_xr, body=None, upper_arms=False):
    """The G1 head, torso and shoulders at the operator (scene frame, G1 scale).

    With PICO body tracking (body: joint name -> XR position), the torso stands at the tracked
    chest and is turned with the tracked shoulders, so turning sideways makes it as thin as the
    robot really is. Without it: a column under the head, turned with the gaze.
    upper_arms: also upper arms towards PICO's elbows. Off for touches: PICO infers the elbows from
    the controllers and its guess flares outwards (measured: all of the extra touches it caused)."""
    head = to_scene(T_scene_from_xr, head_xr[:3, 3])
    top_z = head[2] - (G1_EYE_HEIGHT - G1_SHOULDER_HEIGHT)
    bottom_z = head[2] - (G1_EYE_HEIGHT - G1_PELVIS_HEIGHT)
    centre, lateral = head[:2].copy(), None
    b = {k: to_scene(T_scene_from_xr, np.asarray(v, float)) for k, v in (body or {}).items()
         if np.all(np.isfinite(v))}
    if "spine3" in b:
        centre = b["spine3"][:2]
    if "left_shoulder" in b and "right_shoulder" in b:
        lat = (b["left_shoulder"] - b["right_shoulder"])[:2]
        if np.linalg.norm(lat) > 1e-3:
            lateral = lat / np.linalg.norm(lat)
    if lateral is None:   # gaze yaw: the robot's left is 90 deg to the left of where it looks
        fwd = rot_to_scene(T_scene_from_xr, head_xr[:3, :3]) @ np.array([0, 0, -1.0])
        f = fwd[:2] / max(np.linalg.norm(fwd[:2]), 1e-6)
        lateral = np.array([-f[1], f[0]])
    pts, radii = [head - [0, 0, 0.04]], [HEAD_RADIUS]
    for z in np.linspace(top_z, bottom_z, 5):          # torso + pelvis column
        pts.append(np.r_[centre, z])
        radii.append(G1_TORSO_RADIUS)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        sh = np.r_[centre + sign * G1_SHOULDER_OFFSET * lateral, top_z + 0.01]
        pts.append(sh)
        radii.append(G1_SHOULDER_RADIUS)
        if upper_arms and f"{side}_elbow" in b and f"{side}_shoulder" in b:   # upper arm towards PICO's elbow
            d = b[f"{side}_elbow"] - b[f"{side}_shoulder"]
            if np.linalg.norm(d) > 1e-3:
                d = d / np.linalg.norm(d)
                for t in (0.07, 0.14, 0.21):
                    pts.append(sh + d * t * G1_UPPER_ARM_LENGTH / 0.21)
                    radii.append(G1_UPPER_ARM_RADIUS)
    return np.array(pts), np.array(radii)


def head_in_scene(T_scene_from_xr, head_xr):
    """Head position (G1 scale) and gaze yaw in the scene frame."""
    p = to_scene(T_scene_from_xr, head_xr[:3, 3])
    fwd = rot_to_scene(T_scene_from_xr, head_xr[:3, :3]) @ np.array([0, 0, -1.0])
    return p, float(np.arctan2(fwd[1], fwd[0]))
