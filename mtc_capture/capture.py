"""MTC-style VR capture in the Click-and-Traverse furniture scenes: an operator walks the
scene's route at G1 scale while the headset records head and hand tracking. The aim is to
walk upright and keep the G1's hands and forearms clear of the furniture.

    source ~/Documents/teleop/env.sh
    python -m mtc_capture.capture --operator alice --operator-height 1.74 --generate dense
    python -m mtc_capture.capture --operator alice --operator-height 1.74 --scenes generated_scenes/

Scenes are this branch's cat-furniture-scene-v1 rooms (see furniture.py). Long routes are
captured in segments: windows around each bottleneck, each re-anchored so that it starts on
the operator's home spot.

Open the printed URL in the headset browser (Pico browser / Meta Quest Browser) and press "Virtual Reality". Then:
  1. Stand on your home spot (the end of your free floor), facing along it, upright.
     Calibrate: pinch both hands, held together in front of your face, for 1 s (or type `c`). The room appears
     scaled by 1/alpha so that you are the G1's size, with the segment's start at your feet.
  2. Stay on the start pad facing the blue goal pad (it turns yellow and fills), then
     it turns green: recording.
  3. Follow the teal route to the blue goal pad. Furniture turns orange when the G1-sized
     hand/forearm comes within the margin and red on contact; the goal beacon turns red once
     the take is no longer clean.
  4. Walk back home: the next segment appears (faint until you stand on the pad).
  The same gesture while recording aborts the take.

Keyboard (type + Enter): c calibrate · C calibrate despite the floor check · x abort · d discard last take · n next segment ·
                         N next scene · s status · q quit

Output (raw data, no retargeting):
  <out>/scenes/<scene_id>/scene.json   copy of the scene used
  <out>/takes/<take_id>/motion.npz     per-sample raw XR tracking + clearance annotations
  <out>/takes/<take_id>/meta.json      alpha, transforms, segment, status, safety summary
See mtc_capture/README.md.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
# static files via plain writes: aiohttp's sendfile fallback over TLS raises on dropped
# connections under Python 3.10 (harmless, but floods the terminal)
os.environ.setdefault("AIOHTTP_NOSENDFILE", "1")

import argparse
import base64
import datetime as dt
import json
import math
import queue
import shutil
import sys
import threading
import time

import numpy as np

from . import clearance as C
from . import furniture
from .g1_body import (DEX3_ENVELOPE_END, DEX3_ENVELOPE_RADIUS, DEX3_ENVELOPE_SOURCE, DEX3_ENVELOPE_START,
                      FOREARM_LENGTH, FOREARM_RADIUS, G1_EYE_HEIGHT, G1_HEIGHT, HUMAN_EYE_TO_HEIGHT, TORSO_RADIUS)
from .sources import CTRL_TO_WRIST_OFFSET, WEBXR_JOINTS, FakeSource, NativeSource, XRSource, wrist_from_controller
from .vr_app import box, disc, sphere

VERSION = "mtc_capture/2"
SIDES = ("left", "right")
START_RADIUS = 0.25      # m, G1 scale: head within this of the segment start = on the pad
GESTURE_HOLD = 1.0       # s
COLORS = dict(warn="#ff9f1a", hit="#e8141d", idle="#dddddd", arming="#f5c518", rec="#2ecc71",
              done="#3b82f6", dirty="#e8141d", goal="#3b82f6", route="#11a3a3")


# ----------------------------------------------------------------- scene / segment schedule

class Schedule:
    """Scenes in order; within each scene, its route segments. Segments that already have
    enough successful takes in <out>/takes are skipped, so a collection resumes across sessions."""

    def __init__(self, args, scenes_dir, takes_dir):
        self.args, self.scenes_dir = args, scenes_dir
        self.per_scene_takes = {}
        self.paths = None
        if args.scenes:
            self.paths = []
            for d in args.scenes:
                if d.endswith(".json") and _is_manifest(d):
                    with open(d) as f:
                        m = json.load(f)
                    base = os.path.dirname(os.path.abspath(d))
                    split = {e["path"]: e for e in m["scenes"]}
                    listed = [split[rel] for rel in m["capture_order"] if split[rel]["split"] in args.capture_split]
                    if args.room:
                        listed = select_rooms(listed, args.room)
                    for e in listed:
                        self.paths.append(os.path.join(base, e["path"]))
                        self.per_scene_takes[e["scene_id"]] = e["takes_per_segment"]
                else:
                    self.paths += furniture.find_scenes(d) if os.path.isdir(d) else [d]
            if not self.paths:
                raise SystemExit(f"no cat-furniture-scene-v1 scenes in {args.scenes} (split {args.capture_split})")
        self.done = _existing_takes(takes_dir, args.operator if args.per_operator else None, args.count_only_safe)
        self.scene_i, self.seg_i = 0, 0
        self.scenes_done = 0
        self.finished = False
        self._load()
        self._skip_done()

    def _load(self):
        a = self.args
        if self.paths:
            self.scene = furniture.load(self.paths[self.scene_i])
        else:
            self.scene = furniture.generate(a.generate, a.seed + self.scene_i, a.split)
        self.scene_file = self.scene.save_copy(self.scenes_dir)
        alpha = G1_HEIGHT / a.operator_height
        self.segments = self.scene.all_segments(a.segment_length, a.segments, (alpha, *a.space) if a.space else None,
                                                reverse=a.reverse)
        self.seg_i = 0

    @property
    def segment(self):
        return self.segments[self.seg_i]

    @property
    def needed(self):
        """Takes wanted for the current segment."""
        if self.args.takes_per_segment is not None:
            return self.args.takes_per_segment
        return self.per_scene_takes.get(self.scene.scene_id, 1)

    def key(self):
        return segment_key(self.scene.scene_id, self.segment.s0, self.segment.s1, self.segment.route_id)

    @property
    def done_here(self):
        return self.done[self.key()]

    def record(self):
        self.done[self.key()] += 1

    def _skip_done(self):
        while not self.finished and self.done_here >= self.needed:
            self._step()

    def _step(self):
        if self.seg_i + 1 < len(self.segments):
            self.seg_i += 1
        else:
            self.scenes_done += 1
            self.scene_i += 1
            if self.paths and self.scene_i >= len(self.paths):
                self.finished = True
                return
            self._load()

    def next_segment(self):
        self._step()
        self._skip_done()

    def next_scene(self):
        self.seg_i = len(self.segments) - 1
        self.next_segment()


def room_list(manifest_path, splits=("train", "validation")):
    """Rooms in capture order, numbered from 1 (the numbers --room accepts)."""
    with open(manifest_path) as f:
        m = json.load(f)
    by = {e["path"]: e for e in m["scenes"]}
    return [by[rel] for rel in m["capture_order"] if by[rel]["split"] in splits]


def select_rooms(listed, specs):
    """--room: list numbers ('12', '3-7'), scene ids or parts of them ('chaotic-office-train-000003'),
    or family[:variant] ('chaotic:office', 'hand_table_aisle'). Order as given."""
    out = []
    for spec in specs:
        if spec.replace("-", "").isdigit() and not spec.startswith("-"):
            a, _, b = spec.partition("-")
            picked = listed[int(a) - 1:int(b or a)]
        elif ":" in spec or spec in {e["family"] for e in listed}:
            fam, _, var = spec.partition(":")
            picked = [e for e in listed if e["family"] == fam and (not var or var in (e["variant"], e["role"]))]
        else:
            picked = [e for e in listed if spec in e["scene_id"]]
        if not picked:
            raise SystemExit(f"--room {spec}: no such room (see --list-rooms)")
        out += [e for e in picked if e not in out]
    return out


def print_rooms(args):
    listed = room_list(args.scenes[0], args.capture_split)
    takes_dir = os.path.join(os.path.abspath(args.out), "takes")
    done = _existing_takes(takes_dir, args.operator if args.per_operator else None, False)
    by_scene = {}
    for (sid, *_), n in done.items():
        by_scene[sid] = by_scene.get(sid, 0) + n
    if args.room:
        listed = select_rooms(listed, args.room)
    full = room_list(args.scenes[0], args.capture_split)
    print(f"{'#':>4}  {'family':18s} {'variant':18s} {'split':10s} {'routes':>6s} {'segs':>5s} {'takes done':>10s}  scene_id")
    for e in listed:
        print(f"{full.index(e) + 1:>4}  {e['family']:18s} {str(e['variant'] or e['role']):18s} {e['split']:10s} "
              f"{e.get('routes', 1):>6} {e['segments']:>5} {by_scene.get(e['scene_id'], 0):>10}  {e['scene_id']}")
    print("\nPick with --room: numbers (12, 3-7), family[:variant] (chaotic:office), or scene id text.")


def _is_manifest(path):
    try:
        with open(path) as f:
            return json.load(f).get("schema") == "mtc-capture-scene-set-v1"
    except (OSError, ValueError):
        return False


def _existing_takes(takes_dir, operator, only_safe):
    """segment_key -> number of successful takes already on disk."""
    from collections import Counter
    done = Counter()
    for d in os.listdir(takes_dir) if os.path.isdir(takes_dir) else []:
        try:
            with open(os.path.join(takes_dir, d, "meta.json")) as f:
                m = json.load(f)
        except (OSError, ValueError):
            continue
        if m.get("status") == "success" and (m.get("safe") or not only_safe) and \
                (operator is None or m.get("operator") == operator) and "segment" in m:
            seg = m["segment"]
            done[segment_key(m["scene_id"], seg["route_s0_m"], seg["route_s1_m"], seg.get("route_id", "0"))] += 1
    return done


def segment_key(scene_id, s0, s1, route_id="0"):
    """Segments are identified by their route and route interval (index changes with --space / height)."""
    return scene_id, str(route_id), round(float(s0), 2), round(float(s1), 2)


# ----------------------------------------------------------------- rendering

def scene_elements(scene, seg, ghost=False, highlight=None, state="idle", dirty=False, arm_frac=0.0):
    """Element specs in the scene frame. highlight: asset id -> colour."""
    highlight = highlight or {}
    X, Y, _ = scene.dims
    els = [box("floor", [X / 2, Y / 2, -0.01], [X / 2 + 0.3, Y / 2 + 0.3, 0.01], color="#c9c4b8")]
    for k, b in enumerate(scene.boxes):
        col = highlight.get(scene.asset_of_box[k])
        cat = b.get("category", "")
        op = 0.18 if ghost and cat != "wall" else (0.55 if cat == "wall" else 1.0)
        els.append(box(f"b{k}", b["center"], b["half_size"], b.get("yaw", 0.0),
                       col or furniture.COLORS.get(cat, furniture.DEFAULT_COLOR), op, emissive=col))
    r = seg.route
    for k in range(len(r) - 1):   # the segment's route as a teal strip on the floor
        d = r[k + 1] - r[k]
        L = float(np.linalg.norm(d))
        els.append(box(f"route{k}", [*(r[k] + d / 2), 0.004], [L / 2 + 0.02, 0.02, 0.003],
                       math.atan2(d[1], d[0]), COLORS["route"]))
    pad = {"idle": COLORS["idle"], "arming": COLORS["arming"], "rec": COLORS["rec"]}.get(state, COLORS["idle"])
    els.append(disc("start-pad", [*seg.start[:2], 0.008], START_RADIUS, 0.012, pad, 0.9))
    if state == "arming":   # fills while the countdown runs
        els.append(disc("start-fill", [*seg.start[:2], 0.016], max(0.02, START_RADIUS * arm_frac), 0.012,
                        COLORS["rec"], 0.9))
    els.append(disc("goal-pad", [*seg.goal, 0.008], scene.goal_radius, 0.012, COLORS["goal"], 0.8))
    beacon = COLORS["dirty"] if dirty else {"rec": COLORS["rec"], "arming": COLORS["arming"],
                                            "done": COLORS["done"]}.get(state, COLORS["idle"])
    els.append(sphere("beacon", [*seg.goal, 1.5], 0.08, beacon))
    return els


# ----------------------------------------------------------------- take recorder

class Take:
    def __init__(self, sched, calib, T_xr_from_scene, args, controllers, device=("unknown", "")):
        self.scene, self.seg, self.scene_file = sched.scene, sched.segment, sched.scene_file
        self.calib, self.T, self.args, self.controllers = calib, T_xr_from_scene, args, controllers
        self.device, self.user_agent = device
        self.input_kind = "controllers" if controllers else "hands"
        self.pov = []            # native app: POV video frames (host monotonic s, headset unix ns, JPEG)
        self.native = None   # set by Capture for the native app: wrist offset and the app's hello
        self.t0 = time.monotonic()
        self.wall0 = time.time()
        self.rows = []
        self.hits = {}          # asset id -> set of proxy parts that touched it

    def add(self, s, head_new, clear):
        r = dict(t=s.t - self.t0, head_xr=s.head.copy(), head_new=head_new)
        for side in SIDES:
            r[f"{side}_wrist_xr"] = s.wrist[side].copy()
            st = s.sample_t[side]
            r[f"{side}_sample_t"] = st - self.t0 if np.isfinite(st) else np.nan
            r[f"{side}_tracked"] = s.tracked[side]
            if not self.controllers:
                r[f"{side}_joints_xr"] = np.asarray(s.joints[side]).copy()
                r[f"{side}_joint_rot_xr"] = np.asarray(s.joint_rot[side]).copy()
        r.update(s.buttons)
        r.update(s.extra)
        r.update(clear)
        body_fresh = s.body_age < 0.25 and len(s.body_names) > 0
        r["body_tracked"] = body_fresh
        r["_body"] = dict(zip(s.body_names, s.body.copy())) if body_fresh else {}
        self.rows.append(r)

    def save_pov(self, d):
        """pov.mp4 (the operator's view from the native app) + pov.npz: per frame the time on the take's
        clock (t, like motion.npz) and the headset clock (headset_t, like motion.npz headset_t)."""
        if not self.pov:
            return None
        import cv2
        frames, ts, hts = [], [], []
        for host_t, head_ns, jpg in self.pov:
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
                ts.append(host_t - self.t0)
                hts.append(float(head_ns))
        if not frames:
            return None
        h, w = frames[0].shape[:2]
        fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 15.0
        vw = cv2.VideoWriter(os.path.join(d, "pov.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for img in frames:
            vw.write(img)
        vw.release()
        np.savez_compressed(os.path.join(d, "pov.npz"), t=np.array(ts), headset_t=np.array(hts))
        return dict(file="pov.mp4", frames=len(frames), fps=round(fps, 2), size=[w, h],
                    sync="pov.npz t: take clock (motion.npz t); headset_t: headset unix ns (motion.npz headset_t)")

    def save(self, out_dir, takes_dir, status):
        a = self.args
        stamp = dt.datetime.fromtimestamp(self.wall0).strftime("%Y%m%d-%H%M%S")
        take_id = f"{self.scene.scene_id}__seg{self.seg.index:02d}__{a.operator}__{stamp}"
        d = os.path.join(takes_dir, take_id)
        os.makedirs(d, exist_ok=True)
        arr = {k: np.array([r[k] for r in self.rows]) for k in self.rows[0] if not k.startswith("_")}
        arr["t_unix"] = self.wall0 + arr["t"]
        # body / leg joints (Pico motion trackers or WebXR body tracking), NaN where not reported
        body_names = sorted({n for r in self.rows for n in r["_body"]})
        if body_names:
            body = np.full((len(self.rows), len(body_names), 4, 4), np.nan)
            for i, r in enumerate(self.rows):
                for j, n in enumerate(body_names):
                    if n in r["_body"]:
                        body[i, j] = r["_body"][n]
            arr["body_joints_xr"] = body
        np.savez_compressed(os.path.join(d, "motion.npz"), **arr)
        pov = self.save_pov(d)

        def stat(k):
            v = arr[k][np.isfinite(arr[k])]
            return dict(min=float(v.min()) if len(v) else None, contact_samples=int((v < 0).sum()),
                        warn_samples=int((v < a.hand_margin).sum()), valid_samples=int(len(v)))

        summary = {k: stat(f"clear_{k}") for k in ("left_hand", "right_hand", "left_forearm", "right_forearm", "body")}
        tracked = {side: float(np.mean(arr[f"{side}_tracked"])) for side in SIDES}
        arm_contact = any(summary[k]["contact_samples"] for k in summary if k != "body")
        coverage_ok = min(tracked.values()) >= a.min_tracked
        dur = float(arr["t"][-1])
        Ti = C.invert_similarity(self.T)
        meta = dict(
            take_id=take_id, version=VERSION, status=status,
            safe=bool(status == "success" and not arm_contact and summary["body"]["contact_samples"] == 0 and coverage_ok),
            hands_safe=bool(not arm_contact), tracking_coverage=tracked, tracking_coverage_ok=coverage_ok,
            controller_glitch_fraction={side: float(np.mean(arr[f"{side}_glitch"])) for side in SIDES
                                        if f"{side}_glitch" in arr} or None,
            scene_id=self.scene.scene_id, scene_file=os.path.relpath(self.scene_file, out_dir),
            scene_source=self.scene.source, scene_geometry_hash=self.scene.data.get("geometry_hash"),
            scene_schema=furniture.SCHEMA, segment=self.seg.to_json(),
            operator=a.operator, operator_height=self.calib["operator_height"],
            operator_eye_height=self.calib["eye_height"], alpha=self.calib["alpha"],
            alpha_source=self.calib["alpha_source"],
            T_xr_from_scene=self.T.tolist(), T_scene_from_xr=Ti.tolist(),
            T_xr_from_home=self.calib["T_xr_from_home"].tolist(),
            calibration_time_unix=self.calib["time"], start_unix=self.wall0, duration_s=dur, samples=len(self.rows),
            sample_rate_hz=len(self.rows) / dur if dur > 0 else None,
            input=self.input_kind,
            device=self.device, headset_user_agent=self.user_agent,
            clearance=summary, hand_margin=a.hand_margin,
            assets_touched=[dict(asset_id=i, object=self.scene.labels.get(i, "?"), parts=sorted(p))
                            for i, p in self.hits.items()],
            conventions=dict(
                xr_frame="WebXR local-floor world, OpenXR basis (x right, y up, z back), metres, human scale",
                scene_frame="scene.json frame: metres, z up, origin at the room corner, G1 scale",
                transform="p_scene = T_scene_from_xr @ [p_xr, 1] (rotation, translation, scale alpha); "
                          "T_xr_from_scene = T_xr_from_home @ T_home_from_scene(segment start)",
                poses="4x4 homogeneous matrices, row-major numpy (column vectors)",
                hand_joints="WebXR order (joint_names); positions and rotations in the XR frame",
                wrist="hands: OpenXR wrist joint, +Z towards the elbow, fingers along -Z; "
                      "controllers: grip pose in Unitree arm convention (+x wrist -> fingers); "
                      "native: the OpenXR wrist-joint convention, derived from the controller pose "
                      "({side}_ctrl_xr) with native.wrist_offset (sources.wrist_from_controller)",
                native="native app only: {side}_ctrl_xr raw controller poses, {side}_elbow_xr PICO body "
                       "tracking elbows (nan = none), headset_t headset clock [ns]; body_joints_xr = PICO's "
                       "24-joint skeleton (pico:<joint>, SMPL order), poses in the XR frame",
                time="t: host monotonic s since take start; *_sample_t: time of the XR hand sample "
                     "(same clock; repeated value = no new sample); head_new: head pose changed",
                clearance="clear_*: signed distance [m, G1 scale] of the G1 proxy to the nearest scene box "
                          "(negative = penetration, nan = not tracked). Feedback only; not a substitute for "
                          "checking the retargeted robot."),
            joint_names=WEBXR_JOINTS,
            body_joint_names=body_names,
            body_tracking_coverage=float(np.mean(arr["body_tracked"])) if len(arr["t"]) else 0.0,
            proxy=dict(hand="Dex3 envelope capsule along the hand axis", envelope_source=DEX3_ENVELOPE_SOURCE,
                       envelope_start=DEX3_ENVELOPE_START, envelope_end=DEX3_ENVELOPE_END,
                       envelope_radius=DEX3_ENVELOPE_RADIUS, forearm_length=FOREARM_LENGTH,
                       forearm_radius=FOREARM_RADIUS, torso_radius=TORSO_RADIUS, g1_height=G1_HEIGHT,
                       g1_eye_height=G1_EYE_HEIGHT),
            native=dict(wrist_offset=list(self.native["wrist_offset"]), hello=self.native["hello"])
            if self.native else None,
            pov_video=pov,
            notes=a.notes,
        )
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        return d, meta


# ----------------------------------------------------------------- main loop

class Capture:
    def __init__(self, args):
        self.args = args
        self.out = os.path.abspath(args.out)
        self.takes_dir = os.path.join(self.out, "takes")
        self.discard_dir = os.path.join(self.out, "takes_discarded")
        os.makedirs(self.takes_dir, exist_ok=True)
        self.sched = Schedule(args, os.path.join(self.out, "scenes"), self.takes_dir)
        self.obst = C.ObstacleSet(self.sched.scene)
        if args.fake:
            self.src = FakeSource(careless=args.fake == "careless")
        elif args.native:
            self.src = NativeSource(port=args.native_port, wrist_offset=args.wrist_offset)
        else:
            self.src = XRSource(port=args.port, controllers=args.controllers, show_hands=not args.hide_hands)
        self.controllers = self.src.controllers
        # "hand": s.wrist is a WebXR wrist joint (hand tracking, native app); "controller": a grip pose
        self.wrist_convention = getattr(self.src, "wrist_convention", None) or ("controller" if self.controllers else "hand")
        self.viewer = None   # --fake --view: scene + synthetic operator in a desktop browser
        if args.fake and args.view:
            from .vr_app import CaptureVuer
            self.viewer = CaptureVuer(use_hand_tracking=True, port=args.port, show_hands=False)
        self.show = self.viewer.show if self.viewer else self.src.show
        # operator feedback in the headset: status panel (hud.py) and sound cues (xr_feedback.js)
        self.fx = self.viewer or getattr(self.src, "tv", None)
        self.hud_key, self.refused, self.result, self.result_t = None, None, None, 0.0
        self.hud_png, self.hud_seq = None, 0
        self.facing_ok, self.tick_n, self.hand_lost_t, self.last_sound = True, None, None, {}
        self.calib = None
        self.T = self.Ti = None   # current segment: T_xr_from_scene and inverse
        self.state, self.state_t = "uncalibrated", time.monotonic()
        self.take = None
        self.last_take_dir = None
        self.gesture_since = None
        self.last_head, self.last_head_change = None, time.monotonic()
        self.last_sample = None
        self.highlight = {}
        self.render_key = None
        self.cmds = queue.Queue()
        self.spectator = None   # live state for spectator.py (ZMQ PUB), see publish()
        if args.spectator_port:
            import zmq
            self.spectator = zmq.Context.instance().socket(zmq.PUB)
            self.spectator.setsockopt(zmq.SNDHWM, 2)
            self.spectator.bind(f"tcp://*:{args.spectator_port}")
        self.last_pub = 0.0
        self.n_saved = {"success": 0, "safe": 0, "other": 0}
        self.quit = False
        self.quit_at = None
        self.pending_force = False
        self.view_mode = self.VIEW_MODES.index(getattr(args, "self_view", "both"))   # native: self view (key v / B)
        self.third_person, self.button_down = False, {}
        self.last_body_view = 0.0
        self.pov_n = 0

    @property
    def scene(self):
        return self.sched.scene

    @property
    def seg(self):
        return self.sched.segment

    # ---- calibration / anchoring
    def calibrate(self, head, force=False):
        a = self.args
        eye = float(head[1, 3])
        expect = HUMAN_EYE_TO_HEIGHT * a.operator_height
        if abs(eye - expect) > 0.15 and not force and not a.fake:
            self.refused = dict(eye=eye, exp=expect, off=eye - expect, t=time.monotonic())
            self.sound("refused")
            log(f"[calib] REFUSED: eyes {eye:.2f} m above the headset's floor, expected ~{expect:.2f} m for "
                f"{a.operator_height} m. Stand up straight; if you are, the headset floor is {eye - expect:+.2f} m off: "
                "redo the floor height in the headset's boundary setup (the lobby grid must be at your feet). "
                "Type C + Enter to calibrate anyway.")
            return
        if self.calib is not None and not a.rescale_each_calibration:
            alpha, src, h = self.calib["alpha"], self.calib["alpha_source"], self.calib["operator_height"]
        elif a.operator_height:
            alpha, src, h = G1_HEIGHT / a.operator_height, "height_ratio", a.operator_height
        else:
            alpha, src, h = G1_EYE_HEIGHT / eye, "eye_height_ratio", eye / HUMAN_EYE_TO_HEIGHT
        T_home, yaw = C.home_transform(head, alpha)
        self.calib = dict(alpha=alpha, alpha_source=src, operator_height=h, eye_height=eye,
                          T_xr_from_home=T_home, yaw=yaw, time=time.time())
        self.refused = None
        self.sound("calibrated")
        log(f"[calib] alpha={alpha:.3f} ({src}), eye {eye:.2f} m: the room is shown x{1 / alpha:.2f}; "
            f"headset: {getattr(self.src, 'device', 'unknown')}")
        self.anchor_segment()
        self.set_state("return")

    def anchor_segment(self):
        self.T = self.calib["T_xr_from_home"] @ C.home_from_scene(self.seg.start)
        self.Ti = C.invert_similarity(self.T)
        fwd, width = self.seg.view.floor_need(self.seg.s0, self.seg.s1, self.calib["alpha"])
        warn = ""
        if self.args.space and (fwd > self.args.space[0] + .05 or width > self.args.space[1] + .05):
            warn = f"  WARNING: exceeds --space {self.args.space[0]:g} x {self.args.space[1]:g} m"
        n_routes = len({sg.route_id for sg in self.sched.segments})
        log(f"[segment] {self.scene.scene_id} route {self.seg.route_id} ({n_routes} routes), "
            f"seg {self.seg.index + 1}/{len(self.sched.segments)}: {self.seg.length:.1f} m of route, hazards {[h['id'] for h in self.seg.bottlenecks] or '-'}; "
            f"real floor {fwd:.1f} m ahead x {width:.1f} m wide{warn}")
        if isinstance(self.src, FakeSource):
            self.src.set_path(self.seg.route, self.seg.start[2], self.T)

    def set_state(self, s):
        if s != self.state and s in ("arming", "recording"):
            self.sound({"arming": "arming", "recording": "start"}[s])
        self.state, self.state_t = s, time.monotonic()
        self.tick_n = None

    def sound(self, name, min_gap=0.0):
        now = time.monotonic()
        if self.fx is None or now - self.last_sound.get(name, -1e9) < min_gap:
            return
        self.last_sound[name] = now
        self.fx.event(name)

    BONES = [("pelvis", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("left_ankle", "left_foot"),
             ("pelvis", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"), ("right_ankle", "right_foot"),
             ("pelvis", "spine1"), ("spine1", "spine2"), ("spine2", "spine3"), ("spine3", "left_collar"),
             ("left_collar", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
             ("spine3", "right_collar"), ("right_collar", "right_shoulder"), ("right_shoulder", "right_elbow"),
             ("right_elbow", "right_wrist")]   # no neck / head: nothing in front of the eyes

    VIEW_MODES = ("skeleton", "robot", "both", "off")
    CLEAR_COLORS = {"ok": "#3aa0ff", "warn": "#ff9f1a", "hit": "#e74c3c"}

    def self_view(self, s, now):
        """Native app, ~20 Hz: how the operator is tracked and checked, drawn around them in the headset.
          skeleton  PICO's body skeleton (orange legs, cyan arms/spine) and the controller wrists with a
                    finger stick, green when tracked, red while the controller glitches
          robot     the G1 collision shapes the touch check uses, at the robot's size: Dex3 hand envelopes,
                    forearms, head, torso, shoulders, upper arms; blue clear, orange within the margin,
                    red touching (only once calibrated)
        Cycle with the right controller's B button or v + Enter."""
        if not isinstance(self.src, NativeSource) or now - self.last_body_view < 1 / 20:
            return
        self.last_body_view = now
        mode = self.VIEW_MODES[self.view_mode]
        from .native_app import segment_element
        els = []
        if mode in ("skeleton", "both"):
            if s.body_age < .25 and len(s.body):
                idx = {n.split(":")[-1]: k for k, n in enumerate(s.body_names)}
                P = s.body[:, :3, 3]
                for k, (a, b) in enumerate(self.BONES):
                    leg = any(w in a + b for w in ("hip", "knee", "ankle", "foot"))
                    e = segment_element(f"bone{k}", P[idx[a]], P[idx[b]], .012, "#ff8c00" if leg else "#22c1e6", .7)
                    if e:
                        els.append(e)
            for side in SIDES:
                ctrl = s.extra.get(f"{side}_ctrl_xr")
                if ctrl is None or not np.all(np.isfinite(ctrl)):
                    continue
                col = "#e74c3c" if s.extra.get(f"{side}_glitch", 0.0) > 0 else "#2ecc71"
                W = wrist_from_controller(ctrl, side, self.src.wrist_offset)
                els.append(dict(tag="Sphere", key=f"wrist_{side}", args=[.035], position=W[:3, 3].tolist(),
                                material=dict(color=col, emissive=True)))
                e = segment_element(f"fingers_{side}", W[:3, 3], W[:3, 3] - W[:3, 2] * .16, .015, col, .9, True)
                if e:
                    els.append(e)
        if mode in ("robot", "both") and self.T is not None:
            margin = self.args.hand_margin
            parts = [(f"{side}_{part}", pr) for side in SIDES if s.tracked.get(side)
                     for part, pr in self.arm_proxy(s, side).items()]
            parts.append(("body", C.body_proxy(self.Ti, s.head, self.body_dict(s))))
            R, t, scale = self.T[:3, :3], self.T[:3, 3], 1.0 / self.calib["alpha"]
            for name, (pts, radii) in parts:
                d, _ = self.obst.min_distance(pts, radii)
                col = self.CLEAR_COLORS["hit" if d < 0 else "warn" if d < margin else "ok"]
                for k, (p, r) in enumerate(zip(pts, radii)):
                    els.append(dict(tag="Sphere", key=f"robot_{name}_{k}", args=[float(r * scale)],
                                    position=(R @ p + t).tolist(), material=dict(color=col, opacity=.45)))
        self.src.tv.xr_markers(els)

    def view_buttons(self, s):
        """Controller buttons (edge-triggered): right B cycles the self view, left Y the third-person panel."""
        for key, act in (("right_bButton", "view"), ("left_bButton", "third")):
            down = s.buttons.get(key, 0.0) > .5
            if down and not self.button_down.get(key):
                self.cmds.put("v" if act == "view" else "t")
            self.button_down[key] = down

    def update_hud(self, s, now):
        """Head-locked status panel; re-rendered only when its content changes."""
        if self.fx is None and self.spectator is None:
            return
        from . import hud
        a, seg = self.args, self.seg
        routes = sorted({sg.route_id for sg in self.sched.segments}, key=lambda r: (len(r), r))
        fmt = dict(route=f"{routes.index(seg.route_id) + 1}/{len(routes)}", seg=seg.index + 1,
                   nseg=len(self.sched.segments), take=min(self.sched.done_here + 1, self.sched.needed),
                   takes=self.sched.needed)
        if self.sched.finished:
            key, color = "finished", "blue"
        elif self.calib is None:
            if self.refused and now - self.refused["t"] < 15:
                key, color = "refused", "red"
                fmt.update({k: self.refused[k] for k in ("eye", "exp", "off")})
            else:
                key, color = "uncal", "grey"
        elif self.state in ("done", "return") and self.result and now - self.result_t < 4:
            key, color = self.result, {"safe": "blue", "unsafe": "orange", "aborted": "grey"}[self.result]
        elif self.state == "return":
            key, color = "ret", "grey"
        elif self.state == "arming":
            n = max(1, math.ceil(self.args.arm_seconds - (now - self.state_t)))
            if not self.facing_ok:
                key, color = "face", "yellow"
            else:
                key, color = "arming", "yellow"
                fmt["n"] = n
                if self.tick_n is not None and n < self.tick_n:
                    self.sound("tick")
                self.tick_n = n
        elif self.state == "recording" and self.take is not None:
            fmt["s"] = now - self.take.t0
            if self.take.hits:
                key, color = "rec_dirty", "red"
                fmt["what"] = ", ".join(sorted({self.scene.pretty(i) for i in self.take.hits}))[:60]
            elif not all(s.tracked.get(h) for h in SIDES):
                key, color = "hands", "orange"
            else:
                key, color = "rec", "green"
        else:
            key, color = "ret", "grey"
        state_key = (key, color, a.hud_lang, tuple(sorted((k, round(v, 2) if isinstance(v, float) else v)
                                                          for k, v in fmt.items() if k != "s")),
                     int(fmt.get("s", 0)))
        if state_key != self.hud_key:
            self.hud_key = state_key
            self.hud_png = hud.render(key, color, a.hud_lang, **fmt)
            self.hud_seq += 1
            if self.fx is not None:
                self.fx.hud(self.hud_png, dict(distance=1.4, height=1.4 * a.hud_height, aspect=4.0, below=1.4 * a.hud_offset))

    # ---- per sample
    def clearances(self, s):
        out = {}
        for side in SIDES:
            for part in ("hand", "forearm"):
                out[f"clear_{side}_{part}"], out[f"near_{side}_{part}"] = np.nan, -1
            if not s.tracked.get(side):
                continue
            prox = self.arm_proxy(s, side)
            for part in ("hand", "forearm"):
                out[f"clear_{side}_{part}"], out[f"near_{side}_{part}"] = self.obst.min_distance(*prox[part])
        out["clear_body"], out["near_body"] = self.obst.min_distance(*C.body_proxy(self.Ti, s.head, self.body_dict(s)))
        return out

    @staticmethod
    def body_dict(s):
        """PICO body joints of a fresh sample (name -> XR position), or None."""
        if s.body_age >= .25 or not len(s.body):
            return None
        return {n.split(":")[-1]: s.body[k, :3, 3] for k, n in enumerate(s.body_names)}

    def arm_proxy(self, s, side):
        if self.wrist_convention == "controller":
            return C.controller_proxy(self.Ti, s.wrist[side])
        return C.hand_proxy(self.Ti, s.wrist[side], s.elbow.get(side))

    GESTURE = dict(pinch=0.03, apart=0.45, below=0.50)   # m: thumb-index tips, wrist-wrist, eyes-wrists

    def gesture_parts(self, s):
        """Measurements behind the calibration / abort gesture (hand tracking), for detection and display."""
        out = {}
        for h in SIDES:
            J = np.asarray(s.joints.get(h, np.zeros((25, 3))))
            out[f"pinch_{h}"] = float(np.linalg.norm(J[4] - J[9])) if s.tracked.get(h) and np.any(J) else None
        if all(s.tracked.get(h) for h in SIDES):
            wl, wr, head = s.wrist["left"][:3, 3], s.wrist["right"][:3, 3], s.head[:3, 3]
            out["apart"] = float(np.linalg.norm(wl - wr))
            out["below"] = float(head[1] - min(wl[1], wr[1]))
        return out

    def gesture(self, s):
        """Both hands pinching (thumb tip to index tip from the tracked joints), held together in front of the
        face. Controllers: both triggers + grips."""
        if self.controllers:
            return all(s.buttons.get(f"{h}_triggerValue", 0) > 0.8 and s.buttons.get(f"{h}_squeezeValue", 0) > 0.8
                       for h in SIDES)
        g, lim = self.gesture_parts(s), self.GESTURE
        if "apart" not in g or None in (g["pinch_left"], g["pinch_right"]):
            return False
        return (g["pinch_left"] < lim["pinch"] and g["pinch_right"] < lim["pinch"]
                and g["apart"] < lim["apart"] and g["below"] < lim["below"])

    def gesture_text(self, s):
        """Compact live check of the gesture, e.g. 'pinch L 2 R 6! cm | apart 30 | below 41 | hold 0.4s'."""
        g, lim = self.gesture_parts(s), self.GESTURE
        f = lambda v, l: "--" if v is None else f"{v * 100:.0f}" + ("" if v < l else "!")
        hold = (f" | hold {time.monotonic() - self.gesture_since:.1f}s"
                if self.gesture_since and self.gesture_since < time.monotonic() else "")
        return (f"pinch L {f(g.get('pinch_left'), lim['pinch'])} R {f(g.get('pinch_right'), lim['pinch'])} cm"
                f" | apart {f(g.get('apart'), lim['apart'])} | below {f(g.get('below'), lim['below'])}{hold}"
                "  (! = not met)")

    def step(self, s):
        now = time.monotonic()
        self.last_sample = s
        if self.sched.finished:   # only the "all done" panel remains until quit_at
            return
        head_new = self.last_head is None or not np.array_equal(s.head, self.last_head)
        if head_new:
            self.last_head, self.last_head_change = s.head.copy(), now
        g = s.head_valid and self.gesture(s)
        if g:
            self.gesture_since, self.gesture_last = self.gesture_since or now, now
        elif self.gesture_since and now - getattr(self, "gesture_last", 0) > 0.3:   # ignore dropouts < 0.3 s
            self.gesture_since = None
        fired = self.gesture_since is not None and now - self.gesture_since > GESTURE_HOLD
        if fired:
            self.gesture_since = now + 1e9   # once per hold
        try:
            cmd = self.cmds.get_nowait()
        except queue.Empty:
            cmd = None

        if cmd == "C":   # force: calibrate now, despite the floor check and even without hand tracking
            tv = getattr(self.src, "tv", None)
            if tv is not None and hasattr(tv, "head_locked") and not tv.head_locked.value:
                tv.force_head.value = True   # accept the head of the connection that streams it (the headset)
                log("[calib] forced: taking the head pose from the headset connection without hand tracking; "
                    "hands will only be recorded once the headset tracks them")
            self.pending_force = True
        if self.pending_force:   # calibrate as soon as a head pose arrives
            if s.head_valid and np.any(s.head[:3, :3]):
                self.pending_force = False
                self.calibrate(s.head, force=True)
            return
        if cmd == "c" or (fired and self.state != "recording") or \
                (self.state == "uncalibrated" and self.args.fake and s.head_valid):
            if self.state == "recording":
                self.finish("aborted")
            if s.head_valid:
                self.calibrate(s.head)
            else:
                log("[calib] no headset data yet (no hand tracking?): type C + Enter to calibrate without hands")
            return
        if cmd in ("n", "N") and self.state != "recording":
            self.advance(scene=cmd == "N")
            return
        if cmd == "q":
            self.quit = True
        if cmd == "d":
            self.discard_last()
        if cmd == "s":
            self.print_status(s, force=True)
        if cmd == "v" and isinstance(self.src, NativeSource):
            self.view_mode = (self.view_mode + 1) % len(self.VIEW_MODES)
            if self.VIEW_MODES[self.view_mode] == "off":
                self.src.tv.xr_markers([])
            log(f"[view] self view in the headset: {self.VIEW_MODES[self.view_mode]}")
        if cmd == "t" and isinstance(self.src, NativeSource):
            self.third_person = not self.third_person
            self.src.tv.command("third_person", on=self.third_person)
            log(f"[view] third-person panel {'on' if self.third_person else 'off'}")
        if cmd == "b":
            if isinstance(self.src, NativeSource):
                self.src.tv.command("body_calibrate")
                log("[body] opening the PICO Motion Tracker calibration in the headset")
            else:
                log("[body] b: only with --native")
        if self.state == "uncalibrated" or not s.head_valid:
            return

        p, yaw = C.head_in_scene(self.Ti, s.head)
        on_start = np.linalg.norm(p[:2] - self.seg.start[:2]) < START_RADIUS
        at_goal = np.linalg.norm(p[:2] - self.seg.goal) < self.scene.goal_radius

        if self.state == "return":
            if on_start:
                self.set_state("arming")
        elif self.state == "arming":
            self.facing_ok = abs(math.remainder(yaw - self.seg.start[2], 2 * math.pi)) <= math.radians(self.args.max_start_yaw)
            if not on_start:
                self.set_state("return")
            elif now - self.last_head_change > 0.5:
                self.state_t = now   # head pose frozen (headset off / XR paused): don't start
            elif not self.facing_ok:
                self.state_t = now   # face along the route before the countdown runs
            elif now - self.state_t > self.args.arm_seconds:
                self.take = Take(self.sched, self.calib, self.T, self.args, self.controllers,
                                 (getattr(self.src, "device", "unknown"), getattr(self.src, "headset_user_agent", "")))
                if isinstance(self.src, NativeSource):
                    self.take.input_kind = "native"
                    _, self.pov_n = self.src.tv.pov_since(10 ** 12)
                    self.take.native = dict(wrist_offset=self.src.wrist_offset, hello=self.src.tv.hello())
                self.highlight = {}
                self.set_state("recording")
                if hasattr(self.src, "start_walk"):
                    self.src.start_walk(0.3)
                log(f"[take] recording {self.scene.scene_id} seg {self.seg.index + 1} "
                    f"(take {self.sched.done_here + 1}/{self.sched.needed})")
        elif self.state == "recording":
            last = self.take.rows[-1] if self.take.rows else None
            new_hand = any(np.isfinite(s.sample_t[h]) and (last is None or s.sample_t[h] - self.take.t0 != last[f"{h}_sample_t"])
                           for h in SIDES)
            new_body = s.body_age < 0.25 and (not last or not last["_body"] or
                                              any(not np.array_equal(p, last["_body"].get(nm)) for nm, p in zip(s.body_names, s.body)))
            if head_new or new_hand or new_body:
                clear = self.clearances(s)
                self.take.add(s, head_new, clear)
                self.feedback(clear)
            if not all(s.tracked.get(h) for h in SIDES):   # a hand dropped out of the cameras' view
                self.hand_lost_t = self.hand_lost_t or now
                if now - self.hand_lost_t > 0.4:
                    self.sound("hand_lost", 2.0)
            else:
                self.hand_lost_t = None
            if cmd == "x" or fired:
                self.finish("aborted")
            elif at_goal:
                self.finish("success")
            elif now - self.take.t0 > self.args.max_take_seconds:
                self.finish("timeout")
            elif now - max(self.last_head_change, self.take.t0) > 2.0:   # headset asleep / lost
                self.finish("tracking_lost")
        elif self.state == "done" and now - self.state_t > 1.0:
            self.set_state("return")

    def feedback(self, clear):
        hl = {}
        for side in SIDES:
            for part in ("hand", "forearm"):
                d, i = clear[f"clear_{side}_{part}"], clear[f"near_{side}_{part}"]
                if i >= 0 and np.isfinite(d):
                    if d < 0:
                        if i not in self.take.hits:
                            self.sound("touch", 0.5)
                        self.take.hits.setdefault(i, set()).add(f"{side}_{part}")
                    elif d < self.args.hand_margin:
                        if i not in self.highlight:
                            self.sound("warn", 0.7)
                        hl[i] = COLORS["warn"]
        if np.isfinite(clear["clear_body"]) and clear["clear_body"] < 0 and clear["near_body"] >= 0:
            self.take.hits.setdefault(clear["near_body"], set()).add("body")
        for i in self.take.hits:
            hl[i] = COLORS["hit"]
        self.highlight = hl

    def finish(self, status):
        take, self.take = self.take, None
        self.set_state("done")
        if take is None or not take.rows:
            return
        d, meta = take.save(self.out, self.takes_dir, status)
        self.last_take_dir = d
        self.result = "safe" if status == "success" and meta["safe"] else "unsafe" if status == "success" else "aborted"
        self.result_t = time.monotonic()
        self.sound({"safe": "success", "unsafe": "unsafe", "aborted": "abort"}[self.result])
        cl = meta["clearance"]
        worst = min((v["min"] for k, v in cl.items() if k != "body" and v["min"] is not None), default=float("nan"))
        touched = sorted({f"{t['object']}({','.join(t['parts'])})" for t in meta["assets_touched"]})
        log(f"[take] {status}: {meta['duration_s']:.1f} s, {meta['samples']} samples "
            f"({meta['sample_rate_hz'] or 0:.0f} Hz), tracked L {meta['tracking_coverage']['left']:.0%} "
            f"R {meta['tracking_coverage']['right']:.0%} body {meta['body_tracking_coverage']:.0%} "
            f"({len(meta['body_joint_names'])} joints), min hand/forearm clearance {worst:+.3f} m, "
            f"touched {touched or 'nothing'} -> {'SAFE' if meta['safe'] else 'not safe'}\n       {d}")
        if status == "success":
            self.n_saved["success"] += 1
            self.n_saved["safe"] += meta["safe"]
            if meta["safe"] or not self.args.count_only_safe:
                self.sched.record()
        else:
            self.n_saved["other"] += 1
        if self.sched.done_here >= self.sched.needed:
            self.advance()
        elif isinstance(self.src, FakeSource):
            self.src.set_path(self.seg.route, self.seg.start[2], self.T)

    def advance(self, scene=False):
        old = self.scene.scene_id
        self.sched.next_scene() if scene else self.sched.next_segment()
        if self.sched.finished:
            log("[done] every scheduled segment has its takes")
            self.quit_at = time.monotonic() + 3.0   # leave the "all done" panel up briefly in the headset
            return
        if self.scene.scene_id != old:
            self.obst = C.ObstacleSet(self.scene)
            self.log_scene()
        self.highlight = {}
        if self.calib is not None:
            self.anchor_segment()
            self.set_state("return")
            self.sound("new_segment")

    def log_scene(self):
        sc = self.scene
        c = sc.data.get("counts", {})
        routes = sorted({sg.route_id for sg in self.sched.segments}, key=lambda r: (len(r), r))
        log(f"[scene] {sc.scene_id}: {c.get('tables', '?')} tables, {c.get('chairs', '?')} chairs, "
            f"{len(sc.boxes)} boxes, {len(routes)} route(s) {routes} -> "
            f"{len(self.sched.segments)} segments of <= {self.args.segment_length:g} m")

    def discard_last(self):
        if not self.last_take_dir or not os.path.exists(self.last_take_dir):
            log("[take] nothing to discard")
            return
        os.makedirs(self.discard_dir, exist_ok=True)
        dst = os.path.join(self.discard_dir, os.path.basename(self.last_take_dir))
        shutil.move(self.last_take_dir, dst)
        log(f"[take] discarded -> {dst}")
        self.last_take_dir = None

    def render(self, now):
        if self.calib is None:
            if self.render_key != ("none",):
                self.show(None, [])
                self.render_key = ("none",)
            return
        state = {"return": "idle", "arming": "arming", "recording": "rec", "done": "done"}[self.state]
        arm = min(1.0, (now - self.state_t) / self.args.arm_seconds) if self.state == "arming" else 0.0
        dirty = self.take is not None and bool(self.take.hits)
        key = (self.scene.scene_id, self.seg.index, self.calib["time"], state, round(arm, 1), dirty,
               tuple(sorted(self.highlight.items())), int(now * 15) if self.viewer else 0)
        if key == self.render_key:
            return
        self.render_key = key
        c = self.calib
        H = C.home_from_scene(self.seg.start)
        anchor = dict(position=c["T_xr_from_home"][:3, 3].tolist(), yaw=c["yaw"], scale=1.0 / c["alpha"],
                      local=dict(position=H[:3, 3].tolist(), yaw=-float(self.seg.start[2])))
        els = scene_elements(self.scene, self.seg, ghost=self.state == "return", highlight=self.highlight,
                             state=state, dirty=dirty, arm_frac=arm)
        if self.viewer and self.last_sample is not None:
            els += self.operator_elements(self.last_sample)
        self.show(anchor, els)

    def operator_elements(self, s):
        """The operator as the clearance check sees it (G1 scale): head, hand envelopes, forearms."""
        els = [box("op-head", C.to_scene(self.Ti, s.head[:3, 3]), [0.07, 0.07, 0.08], color="#222222")]
        for side in SIDES:
            if not s.tracked.get(side):
                continue
            prox = self.arm_proxy(s, side)
            for part, (pts, r) in prox.items():
                d, _ = self.obst.min_distance(pts, r)
                col = COLORS["hit"] if d < 0 else COLORS["warn"] if d < self.args.hand_margin else "#3aa0ff"
                els += [sphere(f"op-{side}-{part}-{k}", p, ri, col, 0.8) for k, (p, ri) in enumerate(zip(pts, r))]
        return els

    def publish(self, s, now):
        """Live state for the spectator view (~30 Hz): room, segment, calibration and the raw tracking."""
        if self.spectator is None or now - self.last_pub < 1 / 30:
            return
        self.last_pub = now
        seg = self.seg
        msg = dict(
            t=now, operator=self.args.operator, state=self.state, calibrated=self.calib is not None,
            scene_file=os.path.abspath(self.sched.scene_file), scene_id=self.scene.scene_id,
            route_id=seg.route_id, seg_index=seg.index, n_segs=len(self.sched.segments),
            seg_route=np.asarray(seg.route).tolist(), seg_start=np.asarray(seg.start).tolist(),
            seg_goal=np.asarray(seg.goal).tolist(), goal_radius=self.scene.goal_radius,
            takes_done=self.sched.done_here, takes_needed=self.sched.needed,
            alpha=self.calib["alpha"] if self.calib else None,
            T_scene_from_xr=self.Ti.tolist() if self.Ti is not None else None,
            head=s.head.tolist(), head_valid=s.head_valid,
            wrists={h: s.wrist[h].tolist() for h in SIDES if h in s.wrist},
            joints={h: np.asarray(s.joints[h]).tolist() for h in SIDES if s.joints.get(h) is not None and s.tracked.get(h)},
            hud_png=base64.b64encode(self.hud_png).decode() if self.hud_png else None, hud_seq=self.hud_seq,
            tracked={h: bool(s.tracked.get(h)) for h in SIDES},
            body={n_: p.tolist() for n_, p in zip(s.body_names, s.body)} if s.body_age < .25 else {},
            highlight={int(k): v for k, v in self.highlight.items()},
            dirty=self.take is not None and bool(self.take.hits),
            recording_s=(now - self.take.t0) if self.take is not None else None,
            saved=self.n_saved, arm_frac=min(1.0, (now - self.state_t) / self.args.arm_seconds) if self.state == "arming" else 0.0)
        try:
            self.spectator.send_string(json.dumps(msg), flags=1)   # zmq.NOBLOCK
        except Exception:
            pass

    def print_status(self, s, force=False):
        if self.calib is None:
            if isinstance(self.src, XRSource) and not self.src.connected():
                msg = "headset not connected: open the URL in the headset browser (Pico / Meta Quest)"
            elif isinstance(self.src, NativeSource):
                f, age, _, _ = self.src.tv.latest()
                body = ((f or {}).get("body") or {}).get("state") or {}
                if not self.src.connected():
                    msg = (f"headset app not connected: open 'XRoboToolkit Voice Beta' in the headset (Library > Unknown sources), wear it "
                           f"(it finds this computer by itself, or set ws://{_lan_ip()}:{self.src.tv.port})")
                else:
                    msg = (f"not calibrated | hold both triggers + grips | eyes {s.head[1, 3]:.2f}"
                           f"/{HUMAN_EYE_TO_HEIGHT * self.args.operator_height:.2f} m | ctrl "
                           + " ".join(f"{h[0].upper()}:{'ok' if s.tracked.get(h) else '--'}" for h in SIDES)
                           + f" | body {'ok' if s.body_age < .25 else body.get('text', '--')} | or c+Enter")
            elif not s.head_valid:
                tv = self.src.tv
                heads, hands = tv.head_events.value, tv.hand_events.value
                if heads and not hands:
                    msg = ("headset sends its head pose but NO HAND TRACKING: put the controllers down (switch them off), "
                           "check hand tracking is on in the headset settings, hold both hands in front of you")
                elif heads and hands:
                    msg = "hand data arrives but no fresh hand sample yet: hold both hands in front of the headset"
                else:
                    msg = ("page open but no tracking: the page is not in VR. Reload it in the headset browser "
                           "(close old tabs) and press 'Virtual Reality'")
            else:
                msg = (f"not calibrated | {self.gesture_text(s)} | eyes {s.head[1, 3]:.2f}"
                       f"/{HUMAN_EYE_TO_HEIGHT * self.args.operator_height:.2f} m | or c+Enter")
        else:
            p, yaw = C.head_in_scene(self.Ti, s.head)
            tr = " ".join(f"{h[0].upper()}:{'ok' if s.tracked.get(h) else '--'}" for h in SIDES)
            if isinstance(self.src, NativeSource):
                tr += f" body:{'ok' if s.body_age < .25 else '--'}"
            msg = (f"{self.state:9s} seg {self.seg.index + 1}/{len(self.sched.segments)} "
                   f"take {self.sched.done_here}/{self.sched.needed} | head ({p[0]:.2f},{p[1]:.2f}) "
                   f"yaw {math.degrees(yaw):+4.0f}° | hands {tr} | saved {self.n_saved['success']} ok / "
                   f"{self.n_saved['safe']} safe / {self.n_saved['other']} other")
        width = max(40, shutil.get_terminal_size((120, 20)).columns - 1)
        print(f"\r{msg[:width]:{width}s}", end="\n" if force else "", flush=True)

    def keyboard(self):
        for line in sys.stdin:
            c = line.strip()[:1]
            if c:
                self.cmds.put(c if c in ("N", "C") else c.lower())
                if c == "q":
                    return

    def run(self):
        a = self.args
        if self.viewer:
            log(f"[view] open https://localhost:{a.port}/?ws=wss://localhost:{a.port} in Safari/Chrome "
                "(accept the certificate); drag to orbit, scroll to zoom")
            time.sleep(2.0)
        if isinstance(self.src, NativeSource):
            ip = _lan_ip()
            print(f"\nIn the headset start the app 'XRoboToolkit Voice Beta' (Library > Unknown sources). It finds this computer by its UDP beacon; "
                  f"otherwise set the host to {ip}:{self.src.tv.port} (see mtc_capture/native/README.md).\n"
                  "Calibrate: stand on the home spot, upright, and hold both triggers + grips for 1 s.  "
                  "b + Enter: calibrate the Motion Trackers.  v (or right B): self view skeleton/robot/both/off.  "
                  "t (or left Y): third-person panel.\n")
        if isinstance(self.src, XRSource):
            ip = _lan_ip()
            print(f"\nIn the headset browser (Pico / Meta Quest) open:  https://{ip}:{a.port}/?ws=wss://{ip}:{a.port}\n"
                  f"(Quest over USB after mtc_capture/quest_usb.sh:  https://localhost:{a.port}/?ws=wss://localhost:{a.port})\n"
                  "then press 'Virtual Reality'.\n")
        if self.sched.finished:
            log("[done] every scheduled segment already has its takes in " + self.takes_dir)
            self.src.close()
            return
        threading.Thread(target=self.keyboard, daemon=True).start()
        self.log_scene()
        next_print = 0.0
        t_end = time.monotonic() + a.duration if a.duration else None
        try:
            while not self.quit and not (self.quit_at and time.monotonic() > self.quit_at):
                s = self.src.read()
                self.step(s)
                now = time.monotonic()
                self.render(now)
                self.update_hud(s, now)
                if isinstance(self.src, NativeSource):
                    self.view_buttons(s)
                self.self_view(s, now)
                if self.take is not None and isinstance(self.src, NativeSource):
                    frames, self.pov_n = self.src.tv.pov_since(self.pov_n)
                    self.take.pov += [f for f in frames if f[0] >= self.take.t0]
                self.publish(s, now)
                if now > next_print:
                    self.print_status(s)
                    next_print = now + 0.5
                if (t_end and now > t_end) or (a.max_scenes and self.sched.scenes_done >= a.max_scenes):
                    break
                time.sleep(1 / 240)
        except KeyboardInterrupt:
            pass
        finally:
            if self.take is not None:
                self.finish("aborted")
            print()
            self.src.close()
            if self.viewer:
                self.viewer.close()


def log(msg):
    """An event line; clears the live status line first."""
    print("\r\033[K" + msg, flush=True)


def _lan_ip():
    """This computer's address on the network the headset uses (macOS and Linux)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))   # no packet is sent; picks the interface of the default route
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--operator", default=None, help="operator id, stored with every take (required to capture)")
    p.add_argument("--operator-height", type=float, default=None,
                   help="operator stature [m]; alpha = 1.32 / height (paper). Also sets how segments are split")
    p.add_argument("--out", default=None,
                   help="dataset root (default data/mtc_capture; with --fake data/mtc_capture/_fake_demo, so synthetic "
                        "takes never mix with real ones)")
    g = p.add_argument_group("scenes (Click-and-Traverse cat-furniture-scene-v1)")
    g.add_argument("--scenes", nargs="+", default=None,
                   help="a scene-set manifest.json (make_scene_set.py), scene.json files, or directories of them")
    g.add_argument("--generate", default="dense:table",
                   help="when no --scenes: dense[:family] | pilot[:family] | open | random[:furniture|generic_clutter] "
                        "| clutter[:family] | chaotic[:medium|hard] | hand_table_aisle[:easy|medium|hard] | hand_shelf_passage[:difficulty]")
    g.add_argument("--seed", type=int, default=0, help="first seed for --generate (scene i uses seed + i)")
    g.add_argument("--split", default="train", choices=["train", "validation", "test"])
    g.add_argument("--segments", default="auto", choices=["auto", "gates", "split", "full"],
                   help="auto: windows around each bottleneck if the scene has any, else consecutive pieces")
    g.add_argument("--segment-length", type=float, default=3.0, help="route window length [m, G1 scale]")
    g.add_argument("--space", type=float, nargs=2, default=[5.5, 3.0], metavar=("FWD", "WIDTH"),
                   help="free real floor from the home spot [m]: forward x width (home centred). Segments that "
                        "need more are split (min 1.2 m of route). Default 5.5 x 3.0")
    g.add_argument("--room", nargs="+", default=None,
                   help="capture only these rooms: list numbers (12, 3-7), family[:variant] (chaotic:office), "
                        "or scene id text; all their routes are walked (see --list-rooms)")
    g.add_argument("--list-rooms", action="store_true", help="print the numbered rooms of the scene set and exit")
    g.add_argument("--reverse", action="store_true",
                   help="also walk every route backwards (more paths per room, e.g. corridors)")
    g.add_argument("--takes-per-segment", type=int, default=None,
                   help="successful takes per segment (default: the manifest's value, else 1)")
    g.add_argument("--capture-split", nargs="+", default=["train", "validation"],
                   help="with a manifest: which splits to capture (test scenes are for policy evaluation)")
    g.add_argument("--per-operator", action="store_true",
                   help="count only this operator's existing takes when resuming (each operator does the full set)")
    g.add_argument("--count-only-safe", action="store_true", help="only safe takes count towards --takes-per-segment")
    g.add_argument("--max-scenes", type=int, default=0, help="stop after this many scenes (0 = no limit)")
    g = p.add_argument_group("take")
    g.add_argument("--hand-margin", type=float, default=0.05, help="warning distance [m, G1 scale]")
    g.add_argument("--arm-seconds", type=float, default=3.0)
    g.add_argument("--max-start-yaw", type=float, default=35.0, help="deg from the route direction to start a take")
    g.add_argument("--max-take-seconds", type=float, default=120.0)
    g.add_argument("--min-tracked", type=float, default=0.9, help="hand-tracked fraction needed for safe=true")
    g.add_argument("--rescale-each-calibration", action="store_true")
    g.add_argument("--notes", default="")
    g = p.add_argument_group("device")
    g.add_argument("--port", type=int, default=8012)
    g.add_argument("--controllers", action="store_true", help="controllers instead of hand tracking (no fingers)")
    g.add_argument("--native", action="store_true",
                   help="the native headset app (XRoboToolkit-MTC): controllers + PICO Motion Tracker legs, "
                        "tracked while the operator looks ahead")
    g.add_argument("--native-port", type=int, default=8013, help="WebSocket port for the native app")
    g.add_argument("--self-view", choices=["skeleton", "robot", "both", "off"], default="both",
                   help="native: how you see yourself in the headset (cycle live with the right B button or v)")
    g.add_argument("--wrist-offset", type=float, nargs=3, default=list(CTRL_TO_WRIST_OFFSET), metavar=("X", "Y", "Z"),
                   help="native: controller -> wrist offset in the controller frame [m] (tracker_trial --native measures it)")
    g.add_argument("--hide-hands", action="store_true", help="do not draw the tracked hands in VR")
    g.add_argument("--hud-lang", choices=["ru", "en"], default="ru", help="language of the status panel in the headset")
    g.add_argument("--hud-offset", type=float, default=0.24, help="status panel: how far below eye level, per metre of distance (it floats 1.4 m ahead)")
    g.add_argument("--hud-height", type=float, default=0.12, help="status panel height per metre of distance (0.12 -> 17 cm tall at 1.4 m)")
    g.add_argument("--fake", nargs="?", const="careful", choices=["careful", "careless"], default=None,
                   help="synthetic operator, no headset (test the pipeline)")
    g.add_argument("--view", action="store_true",
                   help="with --fake: show the scene and the synthetic operator in a desktop browser")
    g.add_argument("--spectator-port", type=int, default=5591,
                   help="publish live state for mtc_capture.spectator on this port (0 = off)")
    g.add_argument("--duration", type=float, default=0, help="stop after N s (tests)")
    args = p.parse_args(argv)
    if (args.room or args.list_rooms) and not args.scenes:
        args.scenes = [os.path.join(here, "..", "data", "mtc_capture", "scene_sets", "messy_v3", "manifest.json")]
    if args.out is None:
        args.out = os.path.join(here, "..", "data", "mtc_capture", "_fake_demo" if args.fake else "")
    if args.list_rooms:
        return print_rooms(args)
    if not args.operator or not args.operator_height:
        p.error("--operator and --operator-height are required to capture")
    Capture(args).run()


if __name__ == "__main__":
    main()
