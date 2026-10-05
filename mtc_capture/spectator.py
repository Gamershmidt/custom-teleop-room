"""Spectator view for a second person: the G1 moving through the current room, live.

The capture publishes its state (room, segment, calibration, raw head/hand/body tracking, hazard
highlights) over ZMQ; this window shows the real Unitree G1 + Dex3 model posed from it:
  - pelvis under the headset (G1 head-camera -> pelvis offset), heading = gaze yaw; crouching lowers it
  - arms: the teleop G1 arm IK (pinocchio/CasADi, ~/Documents/teleop/control/arm_ik.py) from the
    operator's wrists, converted exactly like televuer (OpenXR -> robot basis, Unitree arm convention),
    already at G1 scale (alpha)
  - Dex3 fingers in the branch's fixed stand pose; legs in the nominal stance, Pico tracker /
    body joints drawn as orange markers when the headset provides them
  - the room's boxes (hazard colours from the capture: orange = within the margin, red = touched),
    the current route segment, start/goal pads, and a status overlay
This is a live preview only: nothing is retargeted into the dataset.

    source ~/Documents/teleop/env.sh
    mjpython -m mtc_capture.spectator                        # window (macOS needs mjpython)
    mjpython -m mtc_capture.spectator --host 192.168.1.20    # capture running on another computer
    python -m mtc_capture.spectator --offscreen view.mp4 --seconds 20   # record instead of a window
    python -m mtc_capture.spectator --serve        # operator POV + chase view in any browser: http://<mac-ip>:8120
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # IK: <1 ms per solve instead of ~33 ms

import argparse
import json
import math
import sys
import time

import mujoco
import numpy as np

from . import furniture

TELEOP = os.path.expanduser("~/Documents/teleop")
G1_XML = os.path.join(TELEOP, "sim", "assets", "g1", "g1_29dof_with_hand.xml")
HEAD_IN_PELVIS = np.array([0.054, 0.0, 0.474])   # d435 head camera in the pelvis frame (URDF)
DEX3_STAND = {"left_hand_thumb_1_joint": 1.0472, "right_hand_thumb_1_joint": -1.0472}   # geometry-contract.json

# televuer conventions (tv_wrapper.py)
R_OPENXR_ROBOT = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], float)
T_TO_UNITREE_ARM = {"left": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float),
                    "right": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)}


def rgba(hex_color, a=1.0):
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [a], float)


def rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


class Drawer:
    """Adds geoms to an mjvScene (viewer.user_scn or an offscreen renderer's scene)."""

    def __init__(self, scn):
        self.scn = scn

    def _geom(self, gtype, size, pos, mat, color):
        if self.scn.ngeom >= self.scn.maxgeom:
            return
        mujoco.mjv_initGeom(self.scn.geoms[self.scn.ngeom], gtype, np.asarray(size, float), np.asarray(pos, float),
                            np.asarray(mat, float).reshape(9), np.asarray(color, float))
        self.scn.ngeom += 1

    def box(self, center, half, yaw, color):
        self._geom(mujoco.mjtGeom.mjGEOM_BOX, half, center, rz(yaw), color)

    def disc(self, center, radius, height, color):
        self._geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [radius, height / 2, 0], center, np.eye(3), color)   # (radius, half-height)

    def sphere(self, center, radius, color):
        self._geom(mujoco.mjtGeom.mjGEOM_SPHERE, [radius, 0, 0], center, np.eye(3), color)


class Spectator:
    def __init__(self):
        sys.path.insert(0, os.path.join(TELEOP, "control"))
        from arm_ik import ARM_JOINTS, G1ArmIK
        self.model = mujoco.MjModel.from_xml_path(G1_XML)
        self.data = mujoco.MjData(self.model)
        self.ik = G1ArmIK()
        self.arm_adr = [self.model.joint(n).qposadr[0] for n in ARM_JOINTS]
        for name, q in DEX3_STAND.items():
            try:
                self.data.qpos[self.model.joint(name).qposadr[0]] = q
            except KeyError:
                pass
        self.q_arm = np.zeros(len(ARM_JOINTS))
        self.q_arm[[0, 7]] = 0.2    # arms slightly forward until the first IK solution
        self.pelvis = self.model.body("pelvis").id
        self.scene, self.scene_file, self.msg = None, None, None
        self.robot_xy, self.robot_yaw = np.zeros(2), 0.0
        self.draw_hands = False
        self.hud_img, self.hud_seq = None, -1

    # ---- pose the robot from one capture message
    def update(self, msg):
        self.msg = msg
        if msg.get("scene_file") != self.scene_file and msg.get("scene_file") and os.path.exists(msg["scene_file"]):
            self.scene = furniture.load(msg["scene_file"])
            self.scene_file = msg["scene_file"]
        q = self.data.qpos
        Ti = np.array(msg["T_scene_from_xr"]) if msg.get("T_scene_from_xr") else None
        if Ti is None or not msg.get("head_valid"):
            start = msg.get("seg_start") or [0, 0, 0]   # not calibrated: stand on the segment start
            pos, yaw = np.array([start[0], start[1], 0.793]), start[2]
        else:
            alpha = float(np.cbrt(np.linalg.det(Ti[:3, :3])))
            R_sx = Ti[:3, :3] / alpha
            H = np.array(msg["head"])
            head = Ti[:3, :3] @ H[:3, 3] + Ti[:3, 3]
            fwd = R_sx @ H[:3, :3] @ np.array([0, 0, -1.0])
            yaw = math.atan2(fwd[1], fwd[0])
            pos = head - rz(yaw) @ HEAD_IN_PELVIS
            pos[2] = min(max(pos[2], 0.35), 0.85)
            T_pelvis = np.eye(4)
            T_pelvis[:3, :3], T_pelvis[:3, 3] = rz(yaw), pos
            Tp_inv = np.linalg.inv(T_pelvis)
            targets = {}
            for side in ("left", "right"):
                W = msg.get("wrists", {}).get(side)
                if W is None or not msg.get("tracked", {}).get(side):
                    continue
                W = np.array(W)
                Tw = np.eye(4)
                Tw[:3, :3] = R_sx @ W[:3, :3] @ R_OPENXR_ROBOT @ T_TO_UNITREE_ARM[side]
                Tw[:3, 3] = Ti[:3, :3] @ W[:3, 3] + Ti[:3, 3]
                targets[side] = Tp_inv @ Tw
            if targets:
                cur_l, cur_r = self.ik.fk(self.q_arm)
                sol, _ = self.ik.solve(targets.get("left", cur_l), targets.get("right", cur_r), self.q_arm)
                if sol is not None:
                    if "left" not in targets:
                        sol[:7] = self.q_arm[:7]
                    if "right" not in targets:
                        sol[7:] = self.q_arm[7:]
                    self.q_arm = sol
        q[0:3] = pos
        q[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        q[self.arm_adr] = self.q_arm
        mujoco.mj_forward(self.model, self.data)
        self.robot_xy, self.robot_yaw = pos[:2], yaw

    # ---- room, route and markers
    def draw(self, scn):
        d = Drawer(scn)
        msg, sc = self.msg or {}, self.scene
        if sc is not None:
            X, Y, _ = sc.dims
            d.box([X / 2, Y / 2, -0.008], [X / 2 + .3, Y / 2 + .3, .01], 0, rgba("#d8d4ca"))   # 2 mm above the model's ground
            hl = {int(k): v for k, v in msg.get("highlight", {}).items()}
            for k, b in enumerate(sc.boxes):
                cat = b.get("category", "")
                col = hl.get(sc.asset_of_box[k])
                a = 0.25 if cat == "wall" else 1.0
                d.box(b["center"], b["half_size"], b.get("yaw", 0.0), rgba(col or furniture.COLORS.get(cat, furniture.DEFAULT_COLOR), a))
        route = np.array(msg.get("seg_route") or [])
        for p0, p1 in zip(route[:-1], route[1:]):
            v = p1 - p0
            d.box([*(p0 + v / 2), .007], [np.linalg.norm(v) / 2 + .02, .025, .004], math.atan2(v[1], v[0]), rgba("#11a3a3"))
        state = msg.get("state", "")
        pad = {"arming": "#f5c518", "recording": "#2ecc71"}.get(state, "#dddddd")
        if msg.get("seg_start"):
            d.disc([*msg["seg_start"][:2], .01], .25, .012, rgba(pad, .9))
        if msg.get("seg_goal"):
            d.disc([*msg["seg_goal"], .01], msg.get("goal_radius", .3), .012, rgba("#3b82f6", .8))
            d.sphere([*msg["seg_goal"], 1.5], .08, rgba("#e8141d" if msg.get("dirty") else "#2ecc71" if state == "recording" else "#dddddd"))
        Ti = np.array(msg["T_scene_from_xr"]) if msg.get("T_scene_from_xr") else None
        if Ti is not None and self.draw_hands:   # the operator's tracked hand joints (what they see in VR)
            for side, J in msg.get("joints", {}).items():
                for p in np.asarray(J, float) @ Ti[:3, :3].T + Ti[:3, 3]:
                    d.sphere(p, .008, rgba("#f2c9a0"))
        if Ti is not None:
            for name, M in msg.get("body", {}).items():   # Pico trackers / body joints
                p = Ti[:3, :3] @ np.array(M)[:3, 3] + Ti[:3, 3]
                d.sphere(p, .035, rgba("#ff8c00"))

    def texts(self):
        m = self.msg or {}
        if not m:
            return "waiting for the capture (mtc_capture.capture publishes on the spectator port)", ""
        state = {"uncalibrated": "waiting for calibration", "return": "walk back to the start",
                 "arming": f"on the start pad ({m.get('arm_frac', 0):.0%})",
                 "recording": f"RECORDING {m.get('recording_s') or 0:.1f} s", "done": "take saved"}.get(m.get("state"), m.get("state"))
        tr = " ".join(f"{h[0].upper()}:{'ok' if m.get('tracked', {}).get(h) else '--'}" for h in ("left", "right"))
        left = "\n".join(["state", "room", "route / segment", "takes", "hands", "legs", "saved"])
        right = "\n".join([f"{state}{'  (touched something)' if m.get('dirty') else ''}", m.get("scene_id", "")[:48],
                           f"route {m.get('route_id')}  segment {m.get('seg_index', 0) + 1}/{m.get('n_segs', 0)}",
                           f"{m.get('takes_done', 0)}/{m.get('takes_needed', 0)} for this segment",
                           tr, f"{len(m.get('body', {}))} tracker joints" if m.get("body") else "not tracked",
                           f"{m.get('saved', {}).get('success', 0)} ok / {m.get('saved', {}).get('safe', 0)} safe"])
        return left, right


def subscriber(host, port):
    import zmq
    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)   # only the latest state
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.connect(f"tcp://{host}:{port}")
    return sock


def latest(sock):
    import zmq
    try:
        return json.loads(sock.recv_string(flags=zmq.NOBLOCK))
    except zmq.Again:
        return None


def run_window(sp, sock):
    import mujoco.viewer
    with mujoco.viewer.launch_passive(sp.model, sp.data, show_left_ui=False, show_right_ui=False) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = sp.pelvis
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 3.5, -30, 135
        while v.is_running():
            msg = latest(sock)
            if msg is not None:
                sp.update(msg)
            with v.lock():
                v.user_scn.ngeom = 0
                sp.draw(v.user_scn)
                left, right = sp.texts()
                v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right)])
            v.sync()
            time.sleep(1 / 60)


def run_offscreen(sp, sock, out, seconds, fps=25):
    import shutil
    import subprocess
    from PIL import Image, ImageDraw
    sp.model.vis.global_.offwidth, sp.model.vis.global_.offheight = 1280, 720   # model default is smaller
    r = mujoco.Renderer(sp.model, 720, 1280, max_geom=20000)
    cam = mujoco.MjvCamera()
    cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, sp.pelvis
    cam.distance, cam.elevation = 3.8, -32
    frames, t_end = [], time.monotonic() + seconds
    while time.monotonic() < t_end:
        msg = latest(sock)
        if msg is not None:
            sp.update(msg)
        cam.azimuth = math.degrees(sp.robot_yaw) + 150
        r.update_scene(sp.data, cam)
        sp.draw(r.scene)
        img = Image.fromarray(r.render())
        left, right = sp.texts()
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 620, 190], fill=(255, 255, 255))
        for i, (a, b) in enumerate(zip(left.split("\n"), right.split("\n"))):
            draw.text((20, 18 + 24 * i), f"{a:16s} {b}", fill=(20, 20, 20))
        frames.append(np.asarray(img))
        time.sleep(1 / fps)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found (it is in the teleop env: source ~/Documents/teleop/env.sh)")
    h, w = frames[0].shape[:2]
    enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p"] if not out.endswith(".gif") else []
    proc = subprocess.Popen([ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                             "-r", str(fps), "-i", "-", *enc, out], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f[:, :, :3]).tobytes())
    proc.stdin.close()
    proc.wait()
    print(out, len(frames), "frames")


# XR (y up) -> z-up, human scale: the frame of the pre-calibration lobby
ZUP_FROM_XR = np.eye(4)
ZUP_FROM_XR[:3, :3] = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]).T


def draw_lobby(sp, scn):
    """What the operator sees before calibration (vr_app._lobby): floor grid, green ring, four posts, hands."""
    d, m = Drawer(scn), sp.msg or {}
    to = lambda p: ZUP_FROM_XR[:3, :3] @ np.asarray(p, float)
    d.box([0, 0, -.006], [4, 4, .005], 0, rgba("#6b7280"))
    for k in range(-8, 9):
        c = rgba("#e5e7eb" if k % 2 == 0 else "#9ca3af")
        d.box(to([0, .002, k * .5]), [4, .006, .002], 0, c)
        d.box(to([k * .5, .002, 0]), [.006, 4, .002], 0, c)
    d.disc([0, 0, .006], .3, .01, rgba("#27ae60"))
    for (x, z, c) in ((0, -2.5, "#3b82f6"), (2.5, 0, "#f59e0b"), (-2.5, 0, "#f59e0b"), (0, 2.5, "#9ca3af")):
        d.box(to([x, .85, z]), [.075, .075, .85], 0, rgba(c))
    for J in m.get("joints", {}).values():
        for p in np.asarray(J, float):
            d.sphere(to(p), .01, rgba("#f2c9a0"))


def head_camera(sp, cam):
    """Point a free MuJoCo camera from the operator's eyes along their gaze. After calibration in the
    scene frame (G1 scale); before it in the lobby frame (z-up XR, human scale). None if no headset."""
    m = sp.msg or {}
    if not m.get("head_valid"):
        return None
    lobby = not m.get("T_scene_from_xr")
    Ti, H = (ZUP_FROM_XR if lobby else np.array(m["T_scene_from_xr"])), np.array(m["head"])
    alpha = float(np.cbrt(np.linalg.det(Ti[:3, :3])))
    eye = Ti[:3, :3] @ H[:3, 3] + Ti[:3, 3]
    fwd = (Ti[:3, :3] / alpha) @ H[:3, :3] @ np.array([0, 0, -1.0])
    fwd /= np.linalg.norm(fwd)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 0.5
    cam.lookat[:] = eye + cam.distance * fwd
    cam.azimuth = math.degrees(math.atan2(fwd[1], fwd[0]))
    cam.elevation = math.degrees(math.asin(max(-1.0, min(1.0, fwd[2]))))
    return "lobby" if lobby else "scene"


def overlay_panel(sp, img):
    """The operator's status panel (same PNG as in the headset), bottom centre."""
    import base64
    import io
    from PIL import Image
    m = sp.msg or {}
    if m.get("hud_png") and m.get("hud_seq") != sp.hud_seq:
        sp.hud_img, sp.hud_seq = Image.open(io.BytesIO(base64.b64decode(m["hud_png"]))).convert("RGB"), m.get("hud_seq")
    if sp.hud_img is not None:
        w = int(img.width * .55)
        panel = sp.hud_img.resize((w, w // 4))
        img.paste(panel, ((img.width - w) // 2, img.height - w // 4 - 16))
    return img


class Streams:
    """Latest JPEG per view, served as MJPEG (multipart/x-mixed-replace) to any browser."""

    PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Operator POV</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px -apple-system,sans-serif}
#pov{width:100%;max-height:78vh;object-fit:contain;display:block;background:#000}
#row{display:flex;gap:10px;padding:10px;align-items:flex-start}#chase{width:38%}
</style></head><body><img id="pov" src="/pov.mjpg"><div id="row"><img id="chase" src="/chase.mjpg">
<div>Operator's point of view (rebuilt from the headset's tracking: room, route, hazards, hands, status
panel) and the G1 from behind. Live from the capture; nothing here is recorded.</div></div></body></html>"""

    def __init__(self, port):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.frames, self.cond = {}, threading.Condition()
        streams = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = streams.PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                view = self.path.strip("/").replace(".mjpg", "")
                if view not in ("pov", "chase"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last = None
                try:
                    while True:
                        with streams.cond:
                            streams.cond.wait_for(lambda: streams.frames.get(view) is not last, timeout=2.0)
                            jpg = last = streams.frames.get(view)
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def push(self, view, img, quality=75):
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        with self.cond:
            self.frames[view] = buf.getvalue()
            self.cond.notify_all()


def run_server(sp, sock, port, size=(960, 540), fps=20):
    """Render the operator POV (every frame) and the chase view (every 3rd) and serve them over HTTP."""
    from PIL import Image, ImageDraw
    from .capture import _lan_ip
    sp.model.vis.global_.offwidth, sp.model.vis.global_.offheight = size
    sp.model.vis.global_.fovy = 90.0          # roughly the headset's field of view
    r = mujoco.Renderer(sp.model, size[1], size[0], max_geom=20000)
    pov_opt = mujoco.MjvOption()
    pov_opt.geomgroup[:] = 0                  # hide the robot in the POV: the operator does not see it
    chase = mujoco.MjvCamera()
    chase.type, chase.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, sp.pelvis
    chase.distance, chase.elevation = 3.8, -32
    pov = mujoco.MjvCamera()
    streams = Streams(port)
    print(f"Operator POV: open http://{_lan_ip()}:{port}  (or http://localhost:{port} on this Mac); Ctrl-C to stop")
    n = 0
    while True:
        t = time.monotonic()
        msg = latest(sock)
        if msg is not None:
            sp.update(msg)
        view = head_camera(sp, pov)
        if view:
            sp.draw_hands = True
            r.update_scene(sp.data, pov, pov_opt)
            draw_lobby(sp, r.scene) if view == "lobby" else sp.draw(r.scene)
            img = Image.fromarray(r.render())
        else:
            img = Image.new("RGB", size, (30, 32, 36))
            ImageDraw.Draw(img).text((30, size[1] // 2), "waiting for the headset: open the capture page on it and "
                                     "press Virtual Reality", fill=(220, 220, 220))
        streams.push("pov", overlay_panel(sp, img))
        if n % 3 == 0:
            sp.draw_hands = False
            chase.azimuth = math.degrees(sp.robot_yaw) + 150
            r.update_scene(sp.data, chase)
            sp.draw(r.scene)
            streams.push("chase", Image.fromarray(r.render()), quality=65)
        n += 1
        time.sleep(max(0.0, 1 / fps - (time.monotonic() - t)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="computer running mtc_capture.capture")
    p.add_argument("--port", type=int, default=5591, help="its --spectator-port")
    p.add_argument("--offscreen", default=None, help="write a video (.mp4/.gif) instead of opening a window")
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--serve", type=int, nargs="?", const=8120, default=None,
                   help="stream the operator POV + chase view to browsers on this port (default 8120), no window")
    a = p.parse_args()
    sp, sock = Spectator(), subscriber(a.host, a.port)
    if a.serve:
        try:
            run_server(sp, sock, a.serve)
        except KeyboardInterrupt:
            pass
    elif a.offscreen:
        run_offscreen(sp, sock, a.offscreen, a.seconds)
    else:
        run_window(sp, sock)


if __name__ == "__main__":
    main()
