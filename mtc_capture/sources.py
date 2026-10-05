"""Tracking sources for capture: the headset browser (televuer), the native headset app
(native_app.py: controllers + PICO body tracking) and a synthetic walker for tests.

All return a RawSample with the XR data exactly as the headset reports it: WebXR world
frame (OpenXR basis, y up), metres at human scale, 4x4 poses and 25 hand joints in WebXR
order (0 wrist, 1-4 thumb, 5-9 index, 10-14 middle, 15-19 ring, 20-24 pinky).
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np

TRACKING_TIMEOUT = 0.25   # s without a fresh XR hand sample -> hand not tracked

WEBXR_JOINTS = [
    "wrist",
    "thumb-metacarpal", "thumb-phalanx-proximal", "thumb-phalanx-distal", "thumb-tip",
    "index-finger-metacarpal", "index-finger-phalanx-proximal", "index-finger-phalanx-intermediate",
    "index-finger-phalanx-distal", "index-finger-tip",
    "middle-finger-metacarpal", "middle-finger-phalanx-proximal", "middle-finger-phalanx-intermediate",
    "middle-finger-phalanx-distal", "middle-finger-tip",
    "ring-finger-metacarpal", "ring-finger-phalanx-proximal", "ring-finger-phalanx-intermediate",
    "ring-finger-phalanx-distal", "ring-finger-tip",
    "pinky-finger-metacarpal", "pinky-finger-phalanx-proximal", "pinky-finger-phalanx-intermediate",
    "pinky-finger-phalanx-distal", "pinky-finger-tip",
]


@dataclass
class RawSample:
    t: float                                   # host monotonic time of this poll
    head: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
    head_valid: bool = False
    wrist: dict = field(default_factory=dict)          # side -> (4,4)
    joints: dict = field(default_factory=dict)         # side -> (25,3)   (hands mode)
    joint_rot: dict = field(default_factory=dict)      # side -> (25,3,3) (hands mode)
    sample_t: dict = field(default_factory=dict)       # side -> monotonic time of the XR sample, nan if invalid
    tracked: dict = field(default_factory=dict)        # side -> bool
    buttons: dict = field(default_factory=dict)        # name -> float (pinch/squeeze or controller inputs)
    body_names: list = field(default_factory=list)     # body / tracker joints (xr_body.js), e.g. "body:left-foot-ankle"
    body: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4)))
    body_age: float = float("inf")                     # s since the last body sample
    elbow: dict = field(default_factory=dict)          # side -> (3,) elbow position (native body tracking), if any
    extra: dict = field(default_factory=dict)          # further per-row arrays to record (native: raw controller poses)


class XRSource:
    wrist_convention = None   # set in __init__: "hand" (WebXR wrist joint) or "controller" (grip pose)

    def __init__(self, port=8012, controllers=False, show_hands=True):
        from .vr_app import CaptureVuer
        self.controllers = controllers
        self.wrist_convention = "controller" if controllers else "hand"
        self.tv = CaptureVuer(use_hand_tracking=not controllers, port=port, show_hands=show_hands)
        proc = self.tv.process
        t_end = time.monotonic() + 3.0
        while proc.is_alive() and time.monotonic() < t_end:
            time.sleep(0.1)
        if not proc.is_alive():
            self.tv.close()
            raise SystemExit(f"headset page server could not start on port {port} (in use? "
                             f"`lsof -nP -iTCP:{port} -sTCP:LISTEN`); pick another with --port")

    def show(self, anchor, elements):
        self.tv.show(anchor, elements)

    def read(self):
        tv, now = self.tv, time.monotonic()
        s = RawSample(t=now, head=tv.head_pose)
        s.body_age, s.body_names, s.body = tv.body_sample()
        # hands mode: only once the head comes from the hand-tracking headset (see CaptureVuer.on_cam_move)
        s.head_valid = bool(np.any(s.head[:3, :3])) and (self.controllers or bool(tv.head_locked.value)
                                                          or bool(tv.force_head.value))
        for side in ("left", "right"):
            s.wrist[side] = getattr(tv, f"{side}_arm_pose")
            if self.controllers:
                s.tracked[side] = bool(np.any(s.wrist[side][:3, :3]))
                s.sample_t[side] = now if s.tracked[side] else np.nan
                for k in ("triggerValue", "squeezeValue", "aButton", "bButton"):
                    s.buttons[f"{side}_{k}"] = float(getattr(tv, f"{side}_ctrl_{k}"))
                stick = getattr(tv, f"{side}_ctrl_thumbstickValue")
                s.buttons[f"{side}_thumbstick_x"], s.buttons[f"{side}_thumbstick_y"] = float(stick[0]), float(stick[1])
            else:
                ts = getattr(tv, f"{side}_hand_sample_time_shared").value
                s.sample_t[side] = ts if ts > 0 else np.nan
                s.tracked[side] = ts > 0 and now - ts < TRACKING_TIMEOUT
                s.joints[side] = getattr(tv, f"{side}_hand_positions")
                s.joint_rot[side] = getattr(tv, f"{side}_hand_orientations")
                for k in ("pinch", "pinchValue", "squeeze", "squeezeValue"):
                    s.buttons[f"{side}_{k}"] = float(getattr(tv, f"{side}_hand_{k}"))
        return s

    @property
    def device(self):
        return self.tv.device

    @property
    def headset_user_agent(self):
        return self.tv.headset_user_agent

    def probe_report(self):
        return self.tv.probe_report()

    def connected(self):
        import subprocess
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{self.tv._port}", "-sTCP:ESTABLISHED"],
                             capture_output=True, text=True).stdout
        return any(f":{self.tv._port}->" in line for line in out.splitlines())

    def close(self):
        self.tv.close()


# ----------------------------------------------------------------- native headset app

# Controller -> operator wrist, in the controller frame (OpenXR basis: -z = where the controller
# points, +y = its top). Holding a controller, the line wrist -> knuckles runs along the controller's
# pointing direction, so the WebXR wrist frame (-z to the fingers, +z to the elbow, +y out of the back
# of the hand) is the controller frame turned about z (back of the hand = the controller's outer side)
# and moved back towards the elbow. The Dex3 envelope is symmetric about the finger axis, so the
# clearance only depends on the wrist position and that axis. tracker_trial --native measures the
# offset against PICO body tracking's wrist joint; pass the result as --wrist-offset.
CTRL_TO_WRIST_OFFSET = (0.0, 0.0, 0.08)   # m, right controller's frame; x is mirrored for the left hand
_CTRL_TO_WRIST_R = {"right": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]]),   # columns: wrist x, y, z
                    "left": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])}


def wrist_from_controller(ctrl, side, offset=CTRL_TO_WRIST_OFFSET):
    """Controller pose (4x4, XR frame) -> operator wrist pose in the WebXR wrist-joint convention."""
    W = np.eye(4)
    W[:3, :3] = ctrl[:3, :3] @ _CTRL_TO_WRIST_R[side]
    o = np.asarray(offset, float) * ([-1.0, 1.0, 1.0] if side == "left" else 1.0)
    W[:3, 3] = ctrl[:3, :3] @ o + ctrl[:3, 3]
    return W


# Controller glitches: with its LEDs hidden from the headset cameras (behind the body, low at the
# side, fast head turns) PICO keeps reporting a controller as tracked while it extrapolates; the pose
# then teleports (measured: 20-110 m/s jumps, 1.4 m behind the head) and snaps back. A sample is a
# glitch if it jumps faster than a hand moves since the last good sample, is out of arm's reach of the
# head, or disagrees with PICO's own wrist joint; the hand stays "not tracked" until HOLD after the
# last bad sample. Human scale, metres / seconds.
GLITCH_SPEED = 6.0      # m/s; good takes peak at 3-7 m/s over a frame
GLITCH_SLACK = 0.03     # m of jitter allowed on top
GLITCH_REACH = 1.15     # m from the eyes
GLITCH_PICO = 0.30      # m from PICO's wrist joint (good: 9-14 cm)
GLITCH_HOLD = 0.10      # s


class GlitchFilter:
    """Per-hand glitch detector, fed one headset frame at a time (t in seconds)."""

    def __init__(self):
        self.last = {}          # side -> (t, position) of the last good sample
        self.bad_until = {}

    def __call__(self, side, t, p, head=None, pico=None):
        ok = True
        if head is not None and np.linalg.norm(p - head) > GLITCH_REACH:
            ok = False
        if pico is not None and np.all(np.isfinite(pico)) and np.linalg.norm(p - pico) > GLITCH_PICO:
            ok = False
        last = self.last.get(side)
        if ok and last is not None and 0 < t - last[0] < 0.5:
            ok = np.linalg.norm(p - last[1]) <= GLITCH_SPEED * (t - last[0]) + GLITCH_SLACK
        if not ok:
            self.bad_until[side] = t + GLITCH_HOLD
        glitch = (not ok) or t < self.bad_until.get(side, -1.0)
        if not glitch:
            self.last[side] = (t, p.copy())
        return glitch


def glitch_mask(t, ctrl, head, pico=None, side="left"):
    """Offline: (N,) bool glitch per sample from recorded positions (N,3) (head, PICO wrist may be None)."""
    f = GlitchFilter()
    out = np.zeros(len(t), bool)
    for i in range(len(t)):
        if not np.all(np.isfinite(ctrl[i])):
            continue
        out[i] = f(side, float(t[i]), ctrl[i], None if head is None else head[i], None if pico is None else pico[i])
    return out


class NativeSource:
    """The native headset app (XRoboToolkit client + MTC layer): head, controllers and PICO body tracking.

    Wrists come from the controllers, which the headset tracks wherever the arms are (LEDs seen by the
    side and lower cameras, IMU through short gaps), so the operator can look ahead. They are given in
    the WebXR wrist-joint convention, like hand tracking, so the clearance proxy, the spectator and the
    retargeting treat them the same. Calibration and abort use the controller buttons. Body joints are
    PICO's 24-joint skeleton (named "pico:<joint>"); the elbows also orient the forearm proxy.
    """

    controllers = True             # input: controller buttons, no finger joints
    wrist_convention = "hand"      # s.wrist is a WebXR wrist joint pose

    def __init__(self, port=None, wrist_offset=CTRL_TO_WRIST_OFFSET):
        from .native_app import BODY_JOINTS, PORT, NativeLink
        self.tv = NativeLink(port or PORT)
        self.wrist_offset = tuple(float(v) for v in wrist_offset)
        self.body_names = [f"pico:{n}" for n in BODY_JOINTS]
        self._elbow = {side: BODY_JOINTS.index(f"{side}_elbow") for side in ("left", "right")}
        self._wrist = {side: BODY_JOINTS.index(f"{side}_wrist") for side in ("left", "right")}
        self.glitches, self._glitch, self._last_frame = GlitchFilter(), {"left": False, "right": False}, -1

    def show(self, anchor, elements):
        self.tv.show(anchor, elements)

    def read(self):
        from .native_app import pose_to_matrix
        now = time.monotonic()
        f, age, n_frame, t_frame = self.tv.latest()
        s = RawSample(t=now)
        fresh = f is not None and age < TRACKING_TIMEOUT
        s.extra["headset_t"] = float(f.get("t", np.nan)) if f else np.nan
        if f and f.get("head"):
            s.head = pose_to_matrix(f["head"])
            s.head_valid = fresh and bool(f.get("head_ok", True))
        body = (f or {}).get("body") or {}
        joints = body.get("joints")
        if fresh and joints and len(joints) == len(self.body_names):
            s.body_names, s.body, s.body_age = self.body_names, np.array([pose_to_matrix(p) for p in joints]), age
            for side, j in self._elbow.items():
                s.elbow[side] = s.body[j, :3, 3].copy()
        new_frame = n_frame != self._last_frame
        self._last_frame = n_frame
        for side in ("left", "right"):
            c = (f or {}).get("ctrl", {}).get(side) or {}
            ok = fresh and bool(c.get("ok")) and c.get("pose") is not None
            ctrl = pose_to_matrix(c["pose"]) if c.get("pose") is not None else np.full((4, 4), np.nan)
            if ok and new_frame:   # glitch check once per headset frame (read() polls faster than frames arrive)
                pico = s.body[self._wrist[side], :3, 3] if len(s.body) else None
                self._glitch[side] = self.glitches(side, float(f.get("t", now * 1e9)) * 1e-9, ctrl[:3, 3],
                                                   s.head[:3, 3] if s.head_valid else None, pico)
            glitch = ok and self._glitch[side]
            s.extra[f"{side}_ctrl_xr"] = ctrl
            s.extra[f"{side}_glitch"] = float(glitch)
            ok = ok and not glitch                 # a glitching controller is not a tracked hand
            s.wrist[side] = wrist_from_controller(ctrl, side, self.wrist_offset) if ok else np.zeros((4, 4))
            s.tracked[side] = ok
            s.sample_t[side] = t_frame if ok else np.nan
            s.buttons[f"{side}_triggerValue"] = float(c.get("trigger", 0.0))
            s.buttons[f"{side}_squeezeValue"] = float(c.get("grip", 0.0))
            s.buttons[f"{side}_aButton"] = float(c.get("primary", 0.0))
            s.buttons[f"{side}_bButton"] = float(c.get("secondary", 0.0))
            s.buttons[f"{side}_menuButton"] = float(c.get("menu", 0.0))
            stick = c.get("axis") or [0.0, 0.0]
            s.buttons[f"{side}_thumbstick_x"], s.buttons[f"{side}_thumbstick_y"] = float(stick[0]), float(stick[1])
        for side in ("left", "right"):
            s.extra[f"{side}_elbow_xr"] = s.elbow.get(side, np.full(3, np.nan))
        return s

    @property
    def device(self):
        return self.tv.hello().get("device", "native")

    @property
    def headset_user_agent(self):
        h = self.tv.hello()
        return f"native {h.get('app', '?')} {h.get('version', '')} on {h.get('model', '?')}".strip()

    def probe_report(self):
        f, age, n, _ = self.tv.latest()
        return dict(hello=self.tv.hello(), frames=n, frame_age=age, body=(f or {}).get("body", {}).get("state"))

    def connected(self):
        return self.tv.connected()

    def close(self):
        self.tv.close()


# ----------------------------------------------------------------- synthetic walker

def _hand_template(side):
    """25 WebXR joints in the wrist frame (fingers along -z, back of hand +y), open hand."""
    sgn = 1.0 if side == "right" else -1.0     # thumb on +x for the right hand
    pts = [np.zeros(3)]
    fingers = [(0.035, [0.03, 0.065, 0.095, 0.12]),                 # thumb, splayed
               (0.022, [0.035, 0.095, 0.135, 0.16, 0.18]),
               (0.0, [0.035, 0.095, 0.14, 0.168, 0.19]),
               (-0.02, [0.035, 0.09, 0.132, 0.157, 0.177]),
               (-0.038, [0.03, 0.08, 0.11, 0.13, 0.148])]
    for x, zs in fingers:
        for z in zs:
            pts.append(np.array([sgn * x * (1.4 if len(zs) == 4 else 1.0), -0.01, -z]))
    return np.array(pts)


class FakeSource:
    """Synthetic operator for testing without a headset.

    Walks the segment's route at 0.6 m/s (G1 scale) with arms swinging; `careless`
    widens the swing so the hands brush the obstacles. Starts at the XR origin facing -z
    with a 1.65 m eye height and asks the capture loop to calibrate at once. It walks back
    to the start pad instantly after each take.
    """

    def __init__(self, careless=False, eye_height=1.65, speed=0.6):
        self.controllers = False
        self.careless, self.eye_height, self.speed = careless, eye_height, speed
        self.T_xr_from_scene = None
        self.path = None
        self.t0 = time.monotonic()
        self.wait_until = self.t0 + 4.0
        self.hands = {s: _hand_template(s) for s in ("left", "right")}

    def show(self, anchor, elements):
        pass

    def set_path(self, route_xy, start_yaw, T_xr_from_scene):
        """Walk this polyline (scene frame, G1 scale) once start_walk() is called."""
        path = np.asarray(route_xy, float)
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self.path, self.s = path, np.r_[0, np.cumsum(seg)]
        self.start_yaw = start_yaw
        self.T_xr_from_scene = T_xr_from_scene
        self.walk_start = None

    def start_walk(self, delay=0.0):
        self.walk_start = time.monotonic() + delay

    def _head(self, p_scene_xy, heading):
        """Head pose in XR for a G1-scale position/heading in the scene."""
        A = self.T_xr_from_scene[:3, :3]
        scale = np.cbrt(np.linalg.det(A))
        p = self.T_xr_from_scene @ np.r_[p_scene_xy, 0.0, 1.0]
        p = p[:3] + np.array([0, self.eye_height, 0])
        fwd_s = np.array([math.cos(heading), math.sin(heading), 0.0])
        fwd = (A / scale) @ fwd_s
        z = -fwd / np.linalg.norm(fwd)
        y = np.array([0, 1.0, 0])
        x = np.cross(y, z)
        H = np.eye(4)
        H[:3, :3] = np.stack([x, y, z], 1)
        H[:3, 3] = p
        return H

    def read(self):
        now = time.monotonic()
        s = RawSample(t=now)
        if self.T_xr_from_scene is None:   # before calibration: stand at the origin facing -z
            H = np.eye(4)
            H[1, 3] = self.eye_height
        else:
            d = 0.0 if self.walk_start is None else max(0.0, now - self.walk_start) * self.speed
            d = min(d, self.s[-1])
            i = min(np.searchsorted(self.s, d, side="right") - 1, len(self.path) - 2)
            u = (d - self.s[i]) / max(self.s[i + 1] - self.s[i], 1e-9)
            p = self.path[i] + u * (self.path[i + 1] - self.path[i])
            k = max(0, i - 6), min(len(self.path) - 1, i + 6)
            dirv = self.path[k[1]] - self.path[k[0]]
            walking = self.walk_start is not None and now >= self.walk_start
            H = self._head(p, math.atan2(dirv[1], dirv[0]) if walking else self.start_yaw)
        H[:3, 3] += 0.004 * np.array([math.sin(3.1 * now), math.sin(1.7 * now), math.sin(2.3 * now)])  # sway
        s.head, s.head_valid = H, True
        phase = 2 * math.pi * 0.9 * (now - self.t0)
        # synthetic legs (like a tracker/body stream): hips, knees, ankles under the head, stepping
        names, poses = [], []
        for side, sgn in (("left", -1.0), ("right", 1.0)):
            step = 0.12 * math.sin(phase + (0 if side == "left" else math.pi))
            for joint, down, fwd in (("upper-leg", .75, 0.0), ("lower-leg", 1.15, step / 2), ("foot-ankle", 1.55, step)):
                P = np.eye(4)
                P[:3, :3] = H[:3, :3]
                P[:3, 3] = H[:3, :3] @ np.array([sgn * .1, -down, -fwd]) + H[:3, 3]
                names.append(f"body:{side}-{joint}")
                poses.append(P)
        s.body_names, s.body, s.body_age = names, np.array(poses), 0.0
        for side, sgn in (("left", -1.0), ("right", 1.0)):
            swing = (0.35 if self.careless else 0.12) * math.sin(phase + (0 if side == "left" else math.pi))
            out = 0.32 if self.careless else 0.2
            local = np.array([sgn * out, -0.72, -0.08 + swing])        # head frame: x right, y up, z back
            W = np.eye(4)
            W[:3, :3] = H[:3, :3] @ np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])   # fingers down, +z up to the elbow
            W[:3, 3] = H[:3, :3] @ local + H[:3, 3]
            s.wrist[side] = W
            s.joints[side] = self.hands[side] @ W[:3, :3].T + W[:3, 3]
            s.joint_rot[side] = np.repeat(W[None, :3, :3], 25, 0)
            s.sample_t[side] = now
            s.tracked[side] = True
            s.buttons.update({f"{side}_pinch": 0.0, f"{side}_pinchValue": 15.0,
                              f"{side}_squeeze": 0.0, f"{side}_squeezeValue": 0.0})
        return s

    device, headset_user_agent = "fake", "mtc_capture FakeSource"

    def probe_report(self):
        return dict(enabledFeatures=["hand-tracking", "body-tracking (synthetic)"], bodyJoints=["synthetic legs"])

    def connected(self):
        return True

    def close(self):
        pass
