"""Headset side: televuer (Vuer over HTTPS/WebSocket) rendering the capture scene.

The scene is drawn in the headset itself (three.js), not streamed as video, so the
operator can walk through it with full 6-DoF head tracking. Everything sits under one
group placed by the calibration: outer group = home anchor + yaw + scale 1/alpha (XR, y up),
middle group = z-up -> y-up, inner group = scene -> home for the current segment (its start
on the home spot). Scene elements are given in the scene frame (G1 scale).

Vuer's session loop runs in televuer's child process; the capture loop sends it element
specs through a multiprocessing queue. The loop keeps the latest scene and upserts the
whole tree (rate-limited) whenever it changes, so a reconnecting headset gets it too.
"""

import asyncio
import json
import math
import os
import multiprocessing as mp
import queue
import time

import numpy as np
from televuer.televuer import TeleVuer

MAX_BODY_JOINTS = 96
HUD_KEY = "mtc-hud"
TREE_KEY = "mtc-scene"
LOBBY_KEY = "mtc-lobby"
MAX_TREE_RATE = 12.0   # Hz


def box(key, center, half, yaw=0.0, color="#999999", opacity=1.0, emissive=None):
    mat = dict(color=color)
    if opacity < 1.0:
        mat.update(transparent=True, opacity=opacity)
    if emissive:
        mat.update(emissive=emissive, emissiveIntensity=0.6)
    return dict(tag="Box", key=key, args=[2 * half[0], 2 * half[1], 2 * half[2]], position=list(center),
                rotation=[0.0, 0.0, yaw], materialType="standard", material=mat)


def disc(key, center, radius, height, color, opacity=1.0):
    mat = dict(color=color)
    if opacity < 1.0:
        mat.update(transparent=True, opacity=opacity)
    # three.js cylinders run along their local y; turn them onto the scene z axis
    return dict(tag="Cylinder", key=key, args=[radius, radius, height, 48], position=list(center),
                rotation=[math.pi / 2, 0.0, 0.0], materialType="standard", material=mat)


def sphere(key, center, radius, color, opacity=1.0):
    mat = dict(color=color, emissive=color, emissiveIntensity=0.5)
    if opacity < 1.0:
        mat.update(transparent=True, opacity=opacity)
    return dict(tag="Sphere", key=key, args=[radius, 24, 16], position=list(center),
                materialType="standard", material=mat)


async def _send(ws, text):
    try:
        await ws.send_str(text)
    except Exception:
        pass


def device_from_user_agent(ua):
    """quest3 / quest3s / quest_pro / quest2 / pico / unknown, from the headset browser's User-Agent."""
    for key, name in (("Quest 3S", "quest3s"), ("Quest 3", "quest3"), ("Quest Pro", "quest_pro"), ("Quest 2", "quest2"),
                      ("Quest", "quest"), ("Pico", "pico"), ("PICO", "pico")):
        if key in ua:
            return name
    return "unknown"


def _lobby():
    """Before calibration, in the raw XR frame (y up, metres, origin where the session started):
    a floor grid at the headset's floor level, a ring at the origin and four posts 2.5 m away.
    If the grid is not at your feet, the Pico floor height is wrong."""
    from vuer.schemas import Box, Cylinder, Group
    mat = lambda c, **kw: dict(materialType="standard", material=dict(color=c, **kw))
    els = [Box(key="lobby-floor", args=[8, 0.01, 8], position=[0, -0.006, 0], **mat("#6b7280"))]
    for k in range(-8, 9):   # 0.5 m grid
        c = "#e5e7eb" if k % 2 == 0 else "#9ca3af"
        els.append(Box(key=f"lobby-gx{k}", args=[8, 0.004, 0.012], position=[0, 0.002, k * 0.5], **mat(c)))
        els.append(Box(key=f"lobby-gz{k}", args=[0.012, 0.004, 8], position=[k * 0.5, 0.002, 0], **mat(c)))
    els.append(Cylinder(key="lobby-ring", args=[0.3, 0.3, 0.01, 48], position=[0, 0.006, 0], **mat("#27ae60")))
    for k, (x, z, c) in enumerate(((0, -2.5, "#3b82f6"), (2.5, 0, "#f59e0b"), (-2.5, 0, "#f59e0b"), (0, 2.5, "#9ca3af"))):
        els.append(Box(key=f"lobby-post{k}", args=[0.15, 1.7, 0.15], position=[x, 0.85, z], **mat(c)))
    return Group(*els, key=LOBBY_KEY)


def _element(spec):
    from vuer.schemas import Box, Cylinder, Sphere
    spec = dict(spec)
    cls = {"Box": Box, "Cylinder": Cylinder, "Sphere": Sphere}[spec.pop("tag")]
    return cls(**spec)


class CaptureVuer(TeleVuer):
    """TeleVuer whose session draws the capture scene instead of a camera image."""

    _event_sockets = set()   # child process: headset pages listening for sound cues

    def __init__(self, use_hand_tracking=True, port=8012, hand_fps=60, show_hands=True, **kw):
        self._render_q = mp.get_context("spawn").Queue()
        # set once head poses come from the browser that sends WebXR hand tracking (the headset)
        self.head_locked = mp.get_context("spawn").Value("b", False)
        self.head_events = mp.get_context("spawn").Value("i", 0)   # head poses from any browser (diagnostics)
        # forced mode (capture key C): without hand tracking, take the head from the connection that streams
        # it continuously (an XR headset, 30-90 Hz), not from a desktop browser that only moves its camera now and then
        self.force_head = mp.get_context("spawn").Value("b", False)
        self.hand_events = mp.get_context("spawn").Value("i", 0)   # hand messages from any browser
        self._headset_ua = mp.get_context("spawn").Array("c", 512)   # User-Agent of the XR headset browser
        # body / leg tracking from xr_body.js: latest capability report and latest joint poses
        ctx = mp.get_context("spawn")
        self._probe = ctx.Array("c", 32768)
        self._body_names = ctx.Array("c", 8192)
        self._body_poses = ctx.Array("d", MAX_BODY_JOINTS * 16)
        self._body_count = ctx.Value("i", 0)
        self._body_time = ctx.Value("d", 0.0)
        self._hand_fps = hand_fps
        self._show_hands = show_hands
        super().__init__(use_hand_tracking=use_hand_tracking, binocular=True, img_shape=(480, 1280),
                         display_mode="pass-through", port=port, **kw)

    # ---- called from the capture process
    def show(self, anchor, elements):
        """anchor: dict(position, yaw, scale) or None (not calibrated: hide everything)."""
        self._put(("scene", anchor, elements))

    def hud(self, png_bytes, layout=None):
        """Head-locked status panel (a PNG rendered on the Mac, see hud.py)."""
        self._put(("hud", png_bytes, layout or {}))

    def xr_markers(self, elements):
        """Element specs placed directly in the headset's XR frame (y up, metres), e.g. tracker markers."""
        self._put(("xr", elements))

    def event(self, name):
        """A sound cue in the headset (xr_feedback.js): tick, start, touch, success, ..."""
        self._put(("event", name))

    def _put(self, msg):
        try:
            self._render_q.put_nowait(msg)
        except Exception:
            pass

    def close(self):
        # undelivered updates must not block interpreter exit once the page process is gone
        try:
            self._render_q.cancel_join_thread()
            self._render_q.close()
        except Exception:
            pass
        super().close()

    @property
    def headset_user_agent(self):
        return self._headset_ua.value.decode(errors="replace")

    async def on_hand_move(self, event, session, fps=60):
        self.hand_events.value += 1
        await super().on_hand_move(event, session, fps)

    @property
    def device(self):
        return device_from_user_agent(self.headset_user_agent)

    def _create_vuer(self):
        super()._create_vuer()
        from aiohttp import WSMsgType, web
        index = self.vuer.socket_index
        here = os.path.dirname(__file__)
        body_js = open(os.path.join(here, "xr_body.js")).read() + "\n" + open(os.path.join(here, "xr_feedback.js")).read()

        async def socket_index(request):
            ua = request.headers.get("User-Agent", "")
            if request.headers.get("Upgrade", "").lower() == "websocket" and device_from_user_agent(ua) != "unknown":
                self._headset_ua.value = ua.encode()[:511]   # remember which headset browser connected
            resp = await index(request)
            if isinstance(resp, web.Response) and resp.content_type == "text/html" and resp.text and "<head>" in resp.text:
                resp.text = resp.text.replace("<head>", "<head><script>" + body_js + "</script>", 1)
            return resp

        async def probe(request):   # capability report from xr_body.js, once a second
            self._probe.value = (await request.read())[:32767]
            return web.Response(text="ok")

        async def body_stream(request):   # per-frame body / tracker poses from xr_body.js
            ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024)
            await ws.prepare(request)
            names_now = None
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    joints = json.loads(msg.data)["joints"]
                except (ValueError, KeyError, TypeError):
                    continue
                names = sorted(joints)[:MAX_BODY_JOINTS]
                flat = [v for n in names for v in (joints[n] if len(joints[n]) == 16 else [np.nan] * 16)]
                with self._body_time.get_lock():
                    if names != names_now:
                        self._body_names.value = json.dumps(names).encode()[:8191]
                        names_now = names
                    self._body_poses[:len(flat)] = flat
                    self._body_count.value = len(names)
                    self._body_time.value = time.monotonic()
            return ws

        self.vuer._add_route("/mtc/probe", probe, method="POST")
        async def events(request):   # sound cues pushed to the headset page (xr_feedback.js)
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            type(self)._event_sockets.add(ws)
            try:
                async for _ in ws:
                    pass
            finally:
                type(self)._event_sockets.discard(ws)
            return ws

        self.vuer._add_route("/mtc/body", body_stream, method="GET")
        async def hud_png(request):   # the current status panel image (the URL changes with every update)
            st = getattr(type(self), "_child_state", None) or {}
            png = (st.get("hud") or (b"", None))[0]
            if not png:
                raise web.HTTPNotFound()
            return web.Response(body=png, content_type="image/png", headers={"Cache-Control": "no-store"})

        self.vuer._add_route("/mtc/events", events, method="GET")
        self.vuer._add_route("/mtc/hud/{name}", hud_png, method="GET")
        self.vuer.socket_index = socket_index

    # ---- body / leg data, read from the capture process
    def body_sample(self):
        """(age_s, joint names, (J,4,4) poses in the XR frame) of the latest body / tracker sample."""
        with self._body_time.get_lock():
            t, n = self._body_time.value, self._body_count.value
            if t <= 0 or n == 0:
                return float("inf"), [], np.zeros((0, 4, 4))
            names = json.loads(self._body_names.value.decode() or "[]")[:n]
            poses = np.array(self._body_poses[:16 * n]).reshape(n, 16).reshape(n, 4, 4, order="F")
        return time.monotonic() - t, names, poses

    def probe_report(self):
        try:
            return json.loads(self._probe.value.decode() or "{}")
        except ValueError:
            return {}

    # ---- runs in the Vuer child process
    def _place_hud(self, session, state, sent_rev, pose):
        """The status panel as an unlit textured board ~1.4 m ahead of the operator, below eye level,
        lazily following their heading. (Vuer's ImageBackground is a scene *background*: with it in the
        scene, the lit objects render black.)"""
        from vuer.schemas import Plane
        png, layout = state["hud"]
        rev = state.get("hud_rev", 0)
        dist, height = layout.get("distance", 1.4), layout.get("height", .2)
        below, aspect = layout.get("below", .3), layout.get("aspect", 4.0)
        H = self.head_pose
        if np.any(H[:3, :3]):
            fwd = -H[:3, 2]
            fwd = np.array([fwd[0], 0.0, fwd[2]])
            fwd = fwd / (np.linalg.norm(fwd) + 1e-9) if np.linalg.norm(fwd) > 1e-6 else np.array([0, 0, -1.0])
            target = H[:3, 3] + dist * fwd + np.array([0, -below, 0])
        else:   # no head pose yet (lobby before hand tracking): ahead of the origin
            fwd, target = np.array([0, 0, -1.0]), np.array([0, 1.6 - below, -dist])
        yaw = math.atan2(-fwd[0], -fwd[2])   # plane normal (+z) towards the operator
        now = time.monotonic()
        if pose is None:
            pose = (target, yaw, 0.0)
        p, y, t_last = pose
        k = 0.25   # smoothing per update
        p = p + k * (target - p)
        y = y + k * math.remainder(yaw - y, 2 * math.pi)
        moved = pose[2] == 0.0 or np.linalg.norm(p - pose[0]) > .01 or abs(math.remainder(y - pose[1], 2 * math.pi)) > .02
        if rev != sent_rev or (moved and now - t_last > 1 / 15):
            session.upsert(Plane(key=HUD_KEY, args=[height * aspect, height], position=p.tolist(), rotation=[0.0, y, 0.0],
                                 materialType="basic", material=dict(map=f"/mtc/hud/{rev}.png", toneMapped=False)))
            return rev, (p, y, now)
        return sent_rev, (p, y, t_last)

    async def on_cam_move(self, event, session, fps=60):
        """Head poses only from the headset: with hand tracking, the session televuer locked for hand
        data (a real WebXR hand stream). A Mac browser showing the same page must not mix its camera
        into the recorded head track."""
        if getattr(session, "CURRENT_WS_ID", None) in self.vuer.ws:
            self.head_events.value += 1
        if self.use_hand_tracking:
            sid = getattr(session, "CURRENT_WS_ID", None)
            if self._hand_tracking_session_id is None or sid != self._hand_tracking_session_id:
                if not self.force_head.value or not self._streams_head(sid):
                    return
            self.head_locked.value = True
        await super().on_cam_move(event, session, fps)

    def _streams_head(self, sid, rate=20.0):
        """True if this connection sent >= rate head poses per second over the last second."""
        stats = self.__dict__.setdefault("_cam_times", {})
        now = time.monotonic()
        times = [t for t in stats.get(sid, []) if now - t < 1.0] + [now]
        stats[sid] = times
        return len(times) >= rate

    async def main_pass_through(self, session):
        from vuer.schemas import (AmbientLight, DirectionalLight, Group, Hands, HemisphereLight, MotionControllers,
                                  Scene)
        tracker = (Hands(stream=True, key="hands", fps=self._hand_fps, hideLeft=not self._show_hands,
                         hideRight=not self._show_hands) if self.use_hand_tracking else
                   MotionControllers(stream=True, key="motionControllers", left=True, right=True))
        # Vuer's scene defaults to frameloop="demand": a static page barely renders in XR, so head
        # poses arrive at a few Hz, hand joints are hardly sampled and the view stays black.
        session.set @ Scene(
            bgChildren=[AmbientLight(key="mtc-ambient", intensity=0.7), HemisphereLight(key="mtc-hemi", intensity=0.6),
                        DirectionalLight(key="mtc-sun", intensity=1.2, position=[2, 5, 1]), tracker],
            frameloop="always", grid=False, background="#3a4048")
        state = getattr(type(self), "_child_state", None)
        if state is None:  # shared by all sessions of this child process
            state = type(self)._child_state = {"rev": 0, "latest": None}
        sent_rev, last_send, lobby, hud_sent, xr_sent = -1, 0.0, False, -1, -1
        hud_pose = None   # smoothed (position, yaw, time of last send)
        while session.CURRENT_WS_ID in self.vuer.ws:   # until this headset disconnects
            try:
                while True:
                    msg = self._render_q.get_nowait()
                    if msg[0] == "scene":
                        state["latest"] = msg[1:]
                        state["rev"] += 1
                    elif msg[0] == "xr":
                        state["xr"] = msg[1]
                        state["xr_rev"] = state.get("xr_rev", 0) + 1
                    elif msg[0] == "hud":
                        state["hud"] = msg[1:]
                        state["hud_rev"] = state.get("hud_rev", 0) + 1
                    elif msg[0] == "event":   # broadcast once, to every headset page listening
                        for ws in list(type(self)._event_sockets):
                            asyncio.ensure_future(_send(ws, json.dumps(dict(event=msg[1], t=time.time()))))
            except queue.Empty:
                pass
            if state.get("xr_rev", 0) != xr_sent and "xr" in state:
                session.upsert(Group(*[_element(e) for e in state["xr"]], key="mtc-xr"))
                xr_sent = state["xr_rev"]
            if state.get("hud"):
                hud_sent, hud_pose = self._place_hud(session, state, hud_sent, hud_pose)
            now = time.monotonic()
            anchor = state["latest"][0] if state["latest"] is not None else None
            if anchor is None and not lobby:   # not calibrated: a floor grid and markers instead of black
                session.remove(TREE_KEY)
                session.upsert(_lobby())
                lobby, sent_rev = True, state["rev"]
            elif anchor is not None and state["rev"] != sent_rev and now - last_send > 1.0 / MAX_TREE_RATE:
                if lobby:
                    session.remove(LOBBY_KEY)
                    lobby = False
                elements = state["latest"][1]
                local = anchor.get("local", dict(position=[0, 0, 0], yaw=0.0))
                session.upsert(Group(
                    Group(Group(*[_element(e) for e in elements], key="mtc-local", position=local["position"],
                                rotation=[0.0, 0.0, local["yaw"]]),
                          key="mtc-zup", rotation=[-math.pi / 2, 0, 0]),
                    key=TREE_KEY, position=anchor["position"], rotation=[0.0, anchor["yaw"], 0.0],
                    scale=anchor["scale"]))
                sent_rev, last_send = state["rev"], now
            await asyncio.sleep(1 / 60)
