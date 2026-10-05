"""A scripted stand-in for the native headset app, to test capture --native without a headset.

Speaks the app's protocol (native_app.py): sends hello and ~72 Hz track frames (head, controllers,
24 PICO body joints), holds both triggers + grips to calibrate, then walks each segment it is shown:
from the start pad along the teal route strips to the goal pad, and back. Checks what it receives
(room matrices, panel PNG, sound cues) and prints a summary.

    python -m mtc_capture.capture --native --operator test --operator-height 1.75 --generate pilot:table \\
        --out /tmp/native_test --duration 60 &
    python -m mtc_capture.native_fake_headset --height 1.75 --seconds 55
"""

import argparse
import asyncio
import base64
import json
import math
import time

import numpy as np

from .g1_body import HUMAN_EYE_TO_HEIGHT
from .native_app import BODY_JOINTS, PORT


def _yaw_quat(theta):
    return [0.0, math.sin(theta / 2), 0.0, math.cos(theta / 2)]


def _heading(fwd):
    """Yaw about +y that turns the head's -z onto the horizontal direction fwd (x, z)."""
    return math.atan2(-fwd[0], -fwd[1])


def _ry(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class Walker:
    def __init__(self, eye):
        self.eye, self.pos, self.theta = eye, np.zeros(2), 0.0   # (x, z) on the floor, facing -z
        self.path, self.k, self.wait_until = [], 0, 0.0
        self.calibrate_until = time.monotonic() + 2.5
        self.seen_key = None

    def on_scene(self, msg):
        if msg.get("lobby"):
            return
        els = {e["k"]: e for e in msg["els"]}
        if "start-pad" not in els or "goal-pad" not in els:
            return
        xz = lambda e: np.array([e["m"][3], e["m"][11]])
        start, goal = xz(els["start-pad"]), xz(els["goal-pad"])
        strips = sorted((k for k in els if k.startswith("route")), key=lambda k: int(k[5:]))
        key = (tuple(np.round(start, 2)), tuple(np.round(goal, 2)))
        if key == self.seen_key:
            return
        self.seen_key = key
        route = [xz(els[k]) for k in strips]
        # the start pad is where the operator stands; walk the strips, then back along them
        out = [start] + route + [goal]
        self.path = out + out[::-1]
        self.k, self.wait_until = 0, time.monotonic() + 4.0   # countdown on the pad
        if strips:
            m = els[strips[0]]["m"]
            self.theta = _heading(np.array([m[0], m[8]]))   # strip's local x = route direction

    def step(self, dt, speed=0.9):
        now = time.monotonic()
        if not self.path or now < self.wait_until:
            return
        target = self.path[self.k]
        d = target - self.pos
        L = float(np.linalg.norm(d))
        if L < 0.03:
            self.k = (self.k + 1) % len(self.path)
            if self.k == len(self.path) // 2:   # at the goal: pause, then walk back
                self.wait_until = now + 1.5
            elif self.k == 0:
                self.wait_until = now + 4.0
            return
        self.pos = self.pos + d / L * min(L, speed * dt)
        if L > 0.1:
            self.theta = _heading(d / L)

    def frame(self, t):
        R = _ry(self.theta)
        head = np.array([self.pos[0], self.eye, self.pos[1]])
        head += 0.004 * np.array([math.sin(3.1 * t), math.sin(1.7 * t), math.sin(2.3 * t)])   # capture waits for a live head
        q = _yaw_quat(self.theta)
        calib = time.monotonic() < self.calibrate_until
        swing = 0.1 * math.sin(2 * math.pi * 0.9 * t)
        ctrl = {}
        for side, sgn in (("left", -1.0), ("right", 1.0)):
            p = head + R @ np.array([sgn * 0.2, -0.65, -0.25 + sgn * swing])
            ctrl[side] = dict(pose=[*p, *q], ok=True, trigger=1.0 if calib else 0.0, grip=1.0 if calib else 0.0,
                              primary=0.0, secondary=0.0, menu=0.0, axis=[0.0, 0.0])
        local = {"pelvis": (0, -0.75, 0), "head": (0, -0.05, 0.05), "neck": (0, -0.2, 0.05)}
        for side, sgn in (("left", -1.0), ("right", 1.0)):
            local.update({f"{side}_hip": (sgn * .1, -.8, 0), f"{side}_knee": (sgn * .1, -1.2, 0),
                          f"{side}_ankle": (sgn * .1, -1.55, 0), f"{side}_foot": (sgn * .1, -1.6, -.1),
                          f"{side}_collar": (sgn * .08, -.25, .05), f"{side}_shoulder": (sgn * .18, -.28, .05),
                          f"{side}_elbow": (sgn * .22, -.55, 0), f"{side}_wrist": (sgn * .2, -.68, -.18 + sgn * swing),
                          f"{side}_hand": (sgn * .2, -.68, -.27 + sgn * swing)})
        local.update({"spine1": (0, -.65, 0), "spine2": (0, -.5, 0), "spine3": (0, -.38, 0)})
        joints = [[*(head + R @ np.array(local[n])), *q] for n in BODY_JOINTS]
        return dict(type="track", t=int(t * 1e9), head=[*head, *q], head_ok=True, ctrl=ctrl,
                    body=dict(state=dict(tracking=True, calibrated=True, text="ok"), joints=joints))


def pov_jpeg(t, w):
    """A 640x480 test image: time, position, a moving bar (to check frame order and sync)."""
    import cv2
    img = np.full((480, 640, 3), 40, np.uint8)
    x = int((t * 200) % 600)
    cv2.rectangle(img, (x, 200), (x + 40, 280), (60, 200, 60), -1)
    cv2.putText(img, f"t {t:6.2f} s  pos {w.pos[0]:+.2f} {w.pos[1]:+.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()


async def run(a):
    import aiohttp
    w = Walker(HUMAN_EYE_TO_HEIGHT * a.height)
    got = dict(scene=0, lobby=0, hud=0, events={}, cmds=[], max_els=0)
    t_end = time.monotonic() + a.seconds
    async with aiohttp.ClientSession() as http:
        for _ in range(50):
            try:
                ws = await http.ws_connect(f"ws://{a.host}:{a.port}/mtc", max_msg_size=0)
                break
            except aiohttp.ClientError:
                await asyncio.sleep(0.2)
        else:
            raise SystemExit(f"no capture listening on ws://{a.host}:{a.port}/mtc")
        await ws.send_str(json.dumps(dict(type="hello", app="native_fake_headset", version="test",
                                          device="fake_native", model="script", origin="floor")))

        async def receive():
            async for m in ws:
                d = json.loads(m.data)
                if d["type"] == "scene":
                    got["lobby" if d["lobby"] else "scene"] += 1
                    got["max_els"] = max(got["max_els"], len(d["els"]))
                    for e in d["els"]:
                        assert len(e["m"]) == 16 and e["s"] in ("box", "cylinder", "sphere", "plane"), e
                    w.on_scene(d)
                elif d["type"] == "hud":
                    assert base64.b64decode(d["png"])[:4] == b"\x89PNG"
                    got["hud"] += 1
                elif d["type"] == "event":
                    got["events"][d["name"]] = got["events"].get(d["name"], 0) + 1
                elif d["type"] == "cmd":
                    got["cmds"].append(d["name"])

        rx = asyncio.ensure_future(receive())
        t0 = last = next_pov = time.monotonic()
        while time.monotonic() < t_end and not rx.done():
            now = time.monotonic()
            w.step(now - last)
            last = now
            frame = w.frame(now - t0)
            await ws.send_str(json.dumps(frame))
            if a.pov and now >= next_pov:   # POV like MtcPov.cs: 'P' + headset ns + JPEG, 15 fps
                next_pov = now + 1 / 15
                await ws.send_bytes(b"P" + int(frame["t"]).to_bytes(8, "little") + pov_jpeg(now - t0, w))
                got["pov_sent"] = got.get("pov_sent", 0) + 1
            await asyncio.sleep(1 / 72)
        rx.cancel()
        await ws.close()
    print(json.dumps(got, indent=1))
    return got


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--height", type=float, default=1.75, help="operator height [m] (sets the eye height)")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--no-pov", dest="pov", action="store_false", help="do not stream POV test frames")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
