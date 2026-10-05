"""Link to the native headset app: the XRoboToolkit Unity client with the MTC layer (native/unity/).

The Pico browser does not pass the PICO Motion Trackers to web pages, and WebXR hand tracking stops
when the hands leave the cameras' view. The native app reads what the PICO SDK gives an app:

  headset -> host   head, both controllers (pose, trigger, grip, buttons, stick) and PICO body
                    tracking (24 joints from the headset, controllers and two ankle Motion Trackers),
                    every frame (72-90 Hz), already in the WebXR frame the takes use (OpenXR basis:
                    x right, y up, z back; metres; floor-level origin; poses as x,y,z,qx,qy,qz,qw)
  host -> headset   the room (each element as a ready 4x4 matrix in that frame, so the app does no
                    anchor maths), the status panel PNG, sound cues and commands

over one plain WebSocket, ws://<host>:8013/mtc. The host also broadcasts a UDP beacon
("MTC_CAPTURE <port>" to port 8014) so the app finds it without typing an address.

NativeLink offers show / hud / xr_markers / event like CaptureVuer, so capture.py drives it the
same way; sources.NativeSource turns its frames into RawSamples.
"""

import asyncio
import base64
import json
import math
import socket
import threading
import time

import numpy as np

PORT = 8013
BEACON_PORT = 8014
FRAME_TIMEOUT = 0.25   # s without a frame -> not tracked
MAX_SCENE_RATE = 15.0  # Hz

# PICO BodyTrackerRole order (PXR_Plugin.cs), the SMPL 24-joint skeleton
BODY_JOINTS = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
    "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]
BODY = {n: i for i, n in enumerate(BODY_JOINTS)}


# ----------------------------------------------------------------- geometry

def pose_to_matrix(p):
    """[x, y, z, qx, qy, qz, qw] -> 4x4."""
    x, y, z, qx, qy, qz, qw = (float(v) for v in p)
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    M = np.eye(4)
    M[:3, :3] = [[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                 [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                 [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]]
    M[:3, 3] = x, y, z
    return M


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def _t(p):
    M = np.eye(4)
    M[:3, 3] = p
    return M


def anchor_matrix(anchor):
    """The transform vr_app.main_pass_through builds from an anchor (three.js groups), as one matrix:
    XR <- T(position) Ry(yaw) S(scale) <- Rx(-pi/2) (z up -> y up) <- T(local.position) Rz(local.yaw) <- scene."""
    local = anchor.get("local", dict(position=[0, 0, 0], yaw=0.0))
    S = np.diag([anchor["scale"]] * 3 + [1.0])
    return (_t(anchor["position"]) @ _ry(anchor["yaw"]) @ S @ _rx(-math.pi / 2)
            @ _t(local["position"]) @ _rz(local["yaw"]))


def native_element(spec, M):
    """A vr_app element spec (three.js tag/args/position/rotation/material) under parent M -> the app's
    format: key, shape, dims, row-major 4x4 matrix in the XR frame, colour, opacity, emissive."""
    rx, ry, rz = spec.get("rotation", [0.0, 0.0, 0.0])
    L = M @ _t(spec.get("position", [0, 0, 0])) @ _rx(rx) @ _ry(ry) @ _rz(rz)   # three.js Euler order XYZ
    a, mat = spec.get("args", []), spec.get("material", {})
    tag = spec["tag"]
    dims = {"Box": lambda: a[:3], "Cylinder": lambda: [a[0], a[2]], "Sphere": lambda: [a[0]],
            "Plane": lambda: a[:2]}[tag]()
    return dict(k=spec["key"], s=tag.lower(), d=[round(float(v), 5) for v in dims],
                m=[round(float(v), 5) for v in L.reshape(-1)], c=mat.get("color", "#999999"),
                o=float(mat.get("opacity", 1.0)), e=bool(mat.get("emissive")))


def segment_element(key, a, b, radius, color, opacity=1.0, emissive=False):
    """A thin box from point a to point b (XR frame), in the app's format."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    y = b - a
    L = float(np.linalg.norm(y))
    if L < 1e-6:
        return None
    y /= L
    x = np.cross(y, [0, 0, 1.0]) if abs(y[2]) < .9 else np.cross(y, [1.0, 0, 0])
    x /= np.linalg.norm(x)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, np.cross(x, y), (a + b) / 2
    return dict(k=key, s="box", d=[2 * radius, L, 2 * radius], m=[round(float(v), 5) for v in M.reshape(-1)],
                c=color, o=float(opacity), e=bool(emissive))


def lobby_elements():
    """vr_app._lobby in the raw XR frame: floor grid at the headset floor, a ring at the origin, posts."""
    m = lambda c: dict(color=c)
    els = [dict(tag="Box", key="lobby-floor", args=[8, 0.01, 8], position=[0, -0.006, 0], material=m("#6b7280"))]
    for k in range(-8, 9):
        c = "#e5e7eb" if k % 2 == 0 else "#9ca3af"
        els.append(dict(tag="Box", key=f"lobby-gx{k}", args=[8, 0.004, 0.012], position=[0, 0.002, k * 0.5], material=m(c)))
        els.append(dict(tag="Box", key=f"lobby-gz{k}", args=[0.012, 0.004, 8], position=[k * 0.5, 0.002, 0], material=m(c)))
    els.append(dict(tag="Cylinder", key="lobby-ring", args=[0.3, 0.3, 0.01, 48], position=[0, 0.006, 0], material=m("#27ae60")))
    for k, (x, z, c) in enumerate(((0, -2.5, "#3b82f6"), (2.5, 0, "#f59e0b"), (-2.5, 0, "#f59e0b"), (0, 2.5, "#9ca3af"))):
        els.append(dict(tag="Box", key=f"lobby-post{k}", args=[0.15, 1.7, 0.15], position=[x, 0.85, z], material=m(c)))
    return els


# ----------------------------------------------------------------- link

class NativeLink:
    """WebSocket server for the native headset app, in a background thread."""

    def __init__(self, port=PORT, beacon=True):
        self.port = port
        self._lock = threading.Lock()
        self._frame, self._frame_t, self._frames = None, 0.0, 0
        self._hello, self._hello_t = {}, 0.0
        self._pov, self._pov_n = [], 0   # POV frames: (host monotonic s, headset unix ns, JPEG bytes); _pov_n counts all
        self._clients = set()
        self._state = {}            # latest scene / xr / hud / ui message, re-sent to every new client
        self._scene_due, self._scene_sent_t = None, 0.0
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(5.0)
        if self._error:
            raise SystemExit(f"native link could not listen on port {port}: {self._error} "
                             f"(in use? `lsof -nP -iTCP:{port} -sTCP:LISTEN`)")
        self._beacon_stop = threading.Event()
        if beacon:
            threading.Thread(target=self._beacon, daemon=True).start()

    # ---- capture-facing API (same as CaptureVuer)
    def show(self, anchor, elements):
        """anchor: dict(position, yaw, scale, local) from capture.render, or None: the lobby."""
        if anchor is None:
            msg = dict(type="scene", lobby=True, els=[native_element(e, np.eye(4)) for e in lobby_elements()])
        else:
            M = anchor_matrix(anchor)
            msg = dict(type="scene", lobby=False, els=[native_element(e, M) for e in elements])
        with self._lock:
            self._scene_due = msg   # rate-limited in the server loop

    def hud(self, png_bytes, layout=None):
        self._set_state("hud", dict(type="hud", png=base64.b64encode(png_bytes).decode(), layout=layout or {}))

    def xr_markers(self, elements):
        """Element specs directly in the XR frame (y up, metres), e.g. body joint markers. Specs that
        already carry a matrix ("m", the app's format, e.g. from segment_element) pass through."""
        self._set_state("xr", dict(type="xr", els=[e if "m" in e else native_element(e, np.eye(4)) for e in elements]))

    def event(self, name):
        self._broadcast(json.dumps(dict(type="event", name=name)))

    def command(self, name, **kw):
        """body_calibrate (opens the PICO Motion Tracker calibration), body_start, ui (hide=bool)."""
        self._broadcast(json.dumps(dict(type="cmd", name=name, **kw)))

    # ---- tracking
    def latest(self):
        """(frame dict or None, age s, frame count, host time the frame arrived)."""
        with self._lock:
            f, t, n = self._frame, self._frame_t, self._frames
        return f, (time.monotonic() - t) if f is not None else float("inf"), n, t

    def pov_since(self, n):
        """POV frames received after frame count n -> (frames, new count). Keeps the last 300 (~20 s)."""
        with self._lock:
            new = self._pov[max(0, len(self._pov) - (self._pov_n - n)):] if self._pov_n > n else []
            return list(new), self._pov_n

    def hello(self):
        with self._lock:
            return dict(self._hello)

    def connected(self):
        return bool(self._clients) and self.latest()[1] < 1.0

    def close(self):
        self._beacon_stop.set()
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            pass

    # ---- internals
    def _set_state(self, key, msg):
        text = json.dumps(msg)
        with self._lock:
            if self._state.get(key) == text:
                return
            self._state[key] = text
        self._broadcast(text)

    def _broadcast(self, text):
        for ws in list(self._clients):
            asyncio.run_coroutine_threadsafe(self._send(ws, text), self._loop)

    @staticmethod
    async def _send(ws, text):
        try:
            await ws.send_str(text)
        except Exception:
            pass

    def _run(self):
        from aiohttp import WSMsgType, web
        asyncio.set_event_loop(self._loop)

        async def handler(request):
            ws = web.WebSocketResponse(heartbeat=5.0, max_msg_size=0)
            await ws.prepare(request)
            self._clients.add(ws)
            with self._lock:   # a (re)connecting app gets the current room, markers, panel and UI state at once
                states = list(self._state.values())
            for text in states:
                await self._send(ws, text)
            try:
                async for m in ws:
                    if m.type == WSMsgType.BINARY and m.data[:1] == b"P" and len(m.data) > 9:   # POV frame
                        with self._lock:
                            self._pov.append((time.monotonic(), int.from_bytes(m.data[1:9], "little"), bytes(m.data[9:])))
                            self._pov_n += 1
                            del self._pov[:-300]
                        continue
                    if m.type != WSMsgType.TEXT:
                        continue
                    try:
                        d = json.loads(m.data)
                    except ValueError:
                        continue
                    if d.get("type") == "track":
                        with self._lock:
                            self._frame, self._frame_t = d, time.monotonic()
                            self._frames += 1
                    elif d.get("type") == "hello":
                        d["peer"] = request.remote
                        with self._lock:
                            self._hello, self._hello_t = d, time.monotonic()
            finally:
                self._clients.discard(ws)
            return ws

        async def scene_pump():   # at most MAX_SCENE_RATE room updates per second, always the latest
            while True:
                await asyncio.sleep(1.0 / 60)
                with self._lock:
                    msg = self._scene_due
                    due = msg is not None and time.monotonic() - self._scene_sent_t >= 1.0 / MAX_SCENE_RATE
                    if due:
                        self._scene_due, self._scene_sent_t = None, time.monotonic()
                if due:
                    text = json.dumps(msg, separators=(",", ":"))
                    with self._lock:
                        self._state["scene"] = text
                    for ws in list(self._clients):
                        await self._send(ws, text)

        try:
            app = web.Application()
            app.router.add_get("/mtc", handler)
            runner = web.AppRunner(app)
            self._loop.run_until_complete(runner.setup())
            self._loop.run_until_complete(web.TCPSite(runner, "0.0.0.0", self.port).start())
        except Exception as e:   # port in use, ...
            self._error = e
            self._ready.set()
            return
        self._loop.create_task(scene_pump())
        self._ready.set()
        self._loop.run_forever()

    def _beacon(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        msg = f"MTC_CAPTURE {self.port}".encode()
        while not self._beacon_stop.wait(1.0):
            try:
                s.sendto(msg, ("255.255.255.255", BEACON_PORT))
            except OSError:
                pass
