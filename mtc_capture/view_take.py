"""Replay recorded takes in 3D with Viser: the scene furniture, the native G1 meshes, and the
tracked hands at G1 scale (Dex3 hand envelope + forearm capsules coloured by the recorded
clearance: blue ok, orange within the margin, red contact), the finger joints, the head and the
body proxy, with a timeline slider and play/pause. The take files are not changed.

    # 1. G1 meshes per scene and the arm retarget per take (branch environment: MuJoCo)
    source .venv/bin/activate && source .env
    python -m mtc_capture.view_take prepare data/mtc_capture/takes/<take_id>
    python -m mtc_capture.view_take prepare --successful          # every safe success

    # 2. serve (viewer environment) and open the printed URL(s)
    .visual-venv/bin/python -m mtc_capture.view_take serve data/mtc_capture/takes/<take_id>
    .visual-venv/bin/python -m mtc_capture.view_take serve --successful   # safe successes, one port per scene

Several takes of one scene share a server: a dropdown picks the take, "Auto-advance" plays them
in segment order (the whole route). The G1 follows the operator's head (position, height by
bending the knees, gaze yaw) and its arms follow the tracked wrists by IK (arm_ik.py); its Dex3
envelopes and arm links are coloured by the real robot geometry's clearance to the furniture.
Native app takes (PICO body tracking): the G1's feet follow the operator's tracked ankles, and
the tracked 24-joint skeleton (G1 scale) can be shown next to it. Other takes: planned steps.
"""

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

from . import clearance as C
from .g1_body import (DEX3_ENVELOPE_END, DEX3_ENVELOPE_RADIUS, DEX3_ENVELOPE_START, FOREARM_LENGTH, FOREARM_RADIUS,
                      G1_EYE_HEIGHT, G1_PELVIS_HEIGHT, G1_SHOULDER_HEIGHT, HEAD_RADIUS, TORSO_RADIUS)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAKES = os.path.join(REPO, "data", "mtc_capture", "takes")
SIDES = ("left", "right")
STATE_COLORS = {"ok": (52, 152, 219), "warn": (255, 159, 26), "contact": (231, 76, 60)}
HAND_COLORS = {"left": (142, 68, 173), "right": (22, 160, 133)}
SEGMENT_COLORS = [(13, 157, 151), (231, 76, 60), (142, 68, 173), (241, 196, 15), (52, 152, 219), (46, 204, 113)]


def load_meta(take):
    with open(os.path.join(take, "meta.json")) as f:
        return json.load(f)


def scene_json_of(take, meta):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(take))), meta["scene_file"])


def scene_source(meta, scene_json):
    """The scene's source directory. Takes store it as an absolute path on the capture computer;
    on another computer, the same path under this repo (from data/ on)."""
    src = meta.get("scene_source")
    if not src:
        return os.path.dirname(scene_json)
    if not os.path.exists(src) and "/data/" in src:
        src = os.path.join(REPO, "data", src.split("/data/", 1)[1])
    if not os.path.exists(src):   # e.g. "generated:pilot:table": use the copy saved with the takes
        return os.path.dirname(scene_json)
    return src if os.path.isdir(src) else os.path.dirname(src)


def bundle_dir(scene_json):
    return os.path.join(os.path.dirname(scene_json), "viewer")


def retarget_path(scene_json, take_id):
    return os.path.join(bundle_dir(scene_json), "takes", take_id + ".npz")


def successful_takes(takes_dir=TAKES):
    """Fully successful takes: status success, safe (no arm/body contact, tracking coverage ok)."""
    out = []
    for d in sorted(os.listdir(takes_dir)):
        p = os.path.join(takes_dir, d)
        try:
            m = load_meta(p)
        except (OSError, ValueError):
            continue
        if m.get("status") == "success" and m.get("safe"):
            out.append(p)
    return out


def pose(x, y, yaw):
    T = np.eye(4)
    T[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    T[:2, 3] = x, y
    return T


def wxyz_from_matrix(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    zz = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x, y, zz = np.copysign(x, R[2, 1] - R[1, 2]), np.copysign(y, R[0, 2] - R[2, 0]), np.copysign(zz, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, zz])
    return q / np.linalg.norm(q)


def z_to(direction):
    """Rotation taking +z onto direction."""
    d = direction / np.linalg.norm(direction)
    v, c = np.cross([0, 0, 1.0], d), d[2]
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1, -1])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def capsule_mesh(length, radius):
    """Capsule along z centred at the origin, surface-to-surface length."""
    import trimesh
    m = trimesh.creation.capsule(height=max(length - 2 * radius, 1e-4), radius=radius, count=[16, 16])
    m.apply_translation(-m.bounds.mean(0))
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32)


def state(d, margin):
    if not np.isfinite(d):
        return None
    return "contact" if d < 0 else "warn" if d < margin else "ok"


class Take:
    """One take's per-frame geometry in the scene frame."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.id = os.path.basename(self.path)
        self.meta = m = load_meta(self.path)
        self.scene_json = scene_json_of(self.path, m)
        z = self.z = dict(np.load(os.path.join(self.path, "motion.npz")))
        Ti = np.array(m["T_scene_from_xr"])
        self.seg, self.margin = m["segment"], m["hand_margin"]
        self.t = z["t"] - z["t"][0]
        self.n = n = len(self.t)
        self.ctrl = m["input"] == "controllers"
        heads, yaws = zip(*(C.head_in_scene(Ti, z["head_xr"][i]) for i in range(n)))
        self.heads, self.yaws = np.array(heads), np.array(yaws)
        self.hands = {}
        for s in SIDES:
            wr = z[f"{s}_wrist_xr"]
            tracked = z[f"{s}_tracked"].astype(bool) & np.isfinite(wr[:, 0, 3])
            wrist, R = np.full((n, 3), np.nan), np.full((n, 3, 3), np.nan)
            for i in np.flatnonzero(tracked):
                wrist[i] = C.to_scene(Ti, wr[i, :3, 3])
                R[i] = C.rot_to_scene(Ti, wr[i, :3, :3])
            joints = C.to_scene(Ti, z[f"{s}_joints_xr"]) if f"{s}_joints_xr" in z else None
            self.hands[s] = dict(tracked=tracked, wrist=wrist, R=R, joints=joints,
                                 clear_hand=z[f"clear_{s}_hand"], near_hand=z[f"near_{s}_hand"],
                                 clear_forearm=z[f"clear_{s}_forearm"], near_forearm=z[f"near_{s}_forearm"])

        # PICO body tracking (native app): 24 joints in the scene frame, G1 scale; NaN where missing
        self.body, self.body_names = None, []
        if "body_joints_xr" in z and m.get("body_joint_names"):
            B = z["body_joints_xr"][:, :, :3, 3]
            self.body = C.to_scene(Ti, B.reshape(-1, 3)).reshape(B.shape)
            self.body_names = [b.split(":")[-1] for b in m["body_joint_names"]]

        # controller glitches (native app): recorded per sample, or detected now for older takes
        self.glitch = {}
        for s in SIDES:
            if f"{s}_glitch" in z:
                self.glitch[s] = z[f"{s}_glitch"].astype(bool)
            elif f"{s}_ctrl_xr" in z:
                from .sources import glitch_mask
                pico = None
                if self.body is not None and f"{s}_wrist" in self.body_names:
                    pico = z["body_joints_xr"][:, self.body_names.index(f"{s}_wrist"), :3, 3]
                self.glitch[s] = glitch_mask(z["t"], z[f"{s}_ctrl_xr"][:, :3, 3], z["head_xr"][:, :3, 3], pico, s)
        for s, g in self.glitch.items():   # a glitching controller is not a tracked hand (as in capture)
            h = self.hands[s]
            h["tracked"] = h["tracked"] & ~g
            h["wrist"][g], h["R"][g] = np.nan, np.nan
        # POV video (native app): frame times on the take clock
        self.pov_file = os.path.join(self.path, "pov.mp4")
        self.pov_t = None
        if os.path.exists(self.pov_file) and os.path.exists(os.path.join(self.path, "pov.npz")):
            self.pov_t = np.load(os.path.join(self.path, "pov.npz"))["t"] - z["t"][0]
        self._pov_frames = None

        p = retarget_path(self.scene_json, self.id)
        self.g1 = dict(np.load(p)) if os.path.exists(p) else None

    def pov_frames(self, width=480):
        """Decoded POV frames (RGB, scaled to width), loaded on first use."""
        if self._pov_frames is None and self.pov_t is not None:
            import cv2
            cap, frames = cv2.VideoCapture(self.pov_file), []
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                h, w = img.shape[:2]
                frames.append(cv2.cvtColor(cv2.resize(img, (width, int(h * width / w))), cv2.COLOR_BGR2RGB))
            self._pov_frames = frames
        return self._pov_frames or []

    def glitch_intervals(self, s):
        g = self.glitch.get(s)
        if g is None or not g.any():
            return []
        edges = np.flatnonzero(np.diff(np.r_[0, g.astype(int), 0]))
        return [(float(self.t[a]), float(self.t[min(b, self.n) - 1])) for a, b in zip(edges[::2], edges[1::2])]

    @property
    def label(self):
        s = self.seg
        return f"seg {s['index']} · route {s['route_id']} · {self.meta['duration_s']:.1f} s · {self.id[-15:]}"

    def summary(self):
        m, s = self.meta, self.seg
        mins = "  \n".join(f"{k.replace('_', ' ')}: {v['min']:+.3f} m ({v['contact_samples']} contact, "
                           f"{v['warn_samples']} warn)" for k, v in m.get("clearance", {}).items()
                           if v.get("min") is not None)
        return (f"**{self.id}**  \nsegment {s['index']} · route {s['route_id']} · "
                f"{s['route_s1_m'] - s['route_s0_m']:.2f} m  \noperator {m['operator']} · {m['input']} · "
                f"{m['duration_s']:.1f} s, {self.n} samples  \nstatus: {m['status']}, "
                f"{'safe' if m.get('safe') else '**not safe**'}  \n\nmin clearance (G1 scale):  \n{mins}"
                + "".join(f"  \n**{s} controller glitches** ({self.glitch[s].mean():.0%} of frames): "
                          + ", ".join(f"{a:.1f}-{b:.1f} s" for a, b in self.glitch_intervals(s)[:8])
                          for s in SIDES if s in self.glitch and self.glitch[s].any())
                + ("  \nPOV video: " + (f"{len(self.pov_t)} frames" if self.pov_t is not None else "none")))


def prepare(takes):
    sys.path.insert(0, REPO)
    from view_furniture import prepare as prepare_bundle
    done = set()
    for take in takes:
        meta = load_meta(take)
        scene_json = scene_json_of(take, meta)
        out = bundle_dir(scene_json)
        if out in done:
            continue
        done.add(out)
        if os.path.exists(os.path.join(out, "manifest.json")):
            print(f"bundle exists: {out}")
        else:
            prepare_bundle(scene_source(meta, scene_json), out)
            print(f"prepared {out}")
    from . import arm_ik
    models = {}
    for take in takes:
        tk = Take(take)
        scene_dir = scene_source(tk.meta, tk.scene_json)
        if tk.scene_json not in models:
            model, q0 = arm_ik.build_model(scene_dir)
            links = arm_ik.export_links(model, os.path.join(bundle_dir(tk.scene_json), "g1_links"))
            models[tk.scene_json] = model, q0, links
        model, q0, links = models[tk.scene_json]
        r = arm_ik.retarget(model, q0, tk, links)
        parent = []   # each link's nearest ancestor among the links (-1 = root): the G1 skeleton
        for name in links:
            b = model.body_parentid[model.body(name).id]
            while b > 0 and model.body(b).name not in links:
                b = model.body_parentid[b]
            parent.append(links.index(model.body(b).name) if b > 0 else -1)
        r["link_parent"] = np.array(parent)
        path = retarget_path(tk.scene_json, tk.id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, links=np.array(links), **r)
        err = " ".join(f"{s} {np.nanmedian(r[s + '_pos_err']) * 100:.1f}/{np.nanmax(r[s + '_pos_err']) * 100:.0f} cm"
                       for s in SIDES)
        print(f"retargeted {tk.id}: wrist error median/max {err}")


def serve(take_paths, port):
    import viser
    import viser.transforms as tf
    from . import furniture

    takes = sorted((Take(p) for p in take_paths), key=lambda k: (k.seg["route_id"], k.seg["route_s0_m"], k.id))
    scene_json = takes[0].scene_json
    assert all(k.scene_json == scene_json for k in takes), "one server per scene"
    bundle = bundle_dir(scene_json)
    if not os.path.exists(os.path.join(bundle, "robot.glb")):
        raise SystemExit(f"no viewer bundle for this scene; run first (branch env):\n"
                         f"  python -m mtc_capture.view_take prepare {takes[0].path}")
    scene = furniture.load(scene_json)
    data = scene.data

    server = viser.ViserServer(host="127.0.0.1", port=port, label="MTC take replay")
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    server.gui.configure_theme(control_layout="floating", control_width="medium", dark_mode=False,
                               show_logo=False, show_share_button=False, brand_color=(16, 114, 122))

    # room (same look as view_furniture.py)
    width, depth, _ = data["room_dimensions"]
    server.scene.add_box("/floor", dimensions=(width + .2, depth + .2, .04), position=(width / 2, depth / 2, -.03),
                         color=(229, 234, 237), receive_shadow=True)
    palette = {"tabletop": (177, 124, 67), "table_leg": (77, 69, 62), "chair_seat": (26, 116, 130),
               "chair_back": (27, 121, 134), "chair_leg": (42, 66, 74), "chair_armrest": (32, 91, 101),
               "wall": (125, 144, 155), "overhead": (228, 161, 62), "item": (217, 83, 79), "plank": (202, 164, 114),
               "crate": (141, 110, 79), "support": (92, 92, 92), "cabinet": (127, 140, 141), "door": (176, 137, 104),
               "partition": (149, 165, 166), "lamp": (244, 227, 161), "plant": (77, 138, 69),
               "shelf_edge": (160, 120, 76)}
    walls = server.scene.add_frame("/walls", show_axes=False)
    for b in data["boxes"]:
        is_wall = b["category"] == "wall"
        server.scene.add_box(f"/{'walls' if is_wall else 'furniture'}/{b['name']}",
                             dimensions=tuple(2 * np.asarray(b["half_size"])), position=tuple(b["center"]),
                             wxyz=tf.SO3.from_z_radians(b.get("yaw", 0.0)).wxyz,
                             color=palette.get(b["category"], (115, 102, 84)),
                             opacity=.15 if is_wall else 1., cast_shadow=not is_wall, receive_shadow=True)

    # every take: its segment route and head path on the floor (faint), the current one highlighted
    def lines(name, pts, color, thickness):
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) > 1:
            return server.scene.add_line_segments(name, np.stack([pts[:-1], pts[1:]], axis=1),
                                                  colors=color, thickness=thickness)
    overview = server.scene.add_frame("/all_takes", show_axes=False)
    for k, tk in enumerate(takes):
        col = SEGMENT_COLORS[tk.seg["index"] % len(SEGMENT_COLORS)]
        route = np.column_stack([np.asarray(tk.seg["route"], float), np.full(len(tk.seg["route"]), .02)])
        lines(f"/all_takes/{k}/route", route, col, .02)
        lines(f"/all_takes/{k}/head", tk.heads * [1, 1, 0] + [0, 0, .03], col, .004)
        server.scene.add_label(f"/all_takes/{k}/label", f"seg {tk.seg['index']}", position=tuple(route[0] + [0, 0, .2]))

    current = {"frame": None}

    def build_current(tk):
        if current["frame"] is not None:
            current["frame"].remove()
        current["frame"] = server.scene.add_frame("/current", show_axes=False)
        route = np.column_stack([np.asarray(tk.seg["route"], float), np.full(len(tk.seg["route"]), .026)])
        lines("/current/route", route, (13, 157, 151), .033)
        server.scene.add_label("/current/start", "START", position=tuple(route[0] + [0, 0, .12]))
        server.scene.add_label("/current/goal", "GOAL", position=tuple(route[-1] + [0, 0, .12]))
        server.scene.add_cylinder("/current/goal_disc", radius=.3, height=.01, color=(59, 130, 246), opacity=.35,
                                  position=tuple(route[-1][:2]) + (.005,), cast_shadow=False)
        trails = server.scene.add_frame("/current/trails", show_axes=False, visible=show_trails.value)
        lines("/current/trails/head", tk.heads, (85, 85, 85), .006)
        for s in SIDES:
            lines(f"/current/trails/{s}_wrist", tk.hands[s]["wrist"], HAND_COLORS[s], .004)
        current["trails"] = trails
        current["hand_trails"] = server.scene.add_frame("/current/hand_trails", show_axes=False,
                                                        visible=show_hl.value)
        if tk.g1 is not None and rig is not None:
            for s in SIDES:                                    # where the G1's hands went
                lines(f"/current/hand_trails/{s}", robot_hand_centres(tk)[s], HAND_COLORS[s], .012)

    # G1: articulated links driven by the arm retarget (arm_ik.py); without one, the baked
    # default pose moved under the head
    links_dir = os.path.join(bundle, "g1_links")
    rig = json.load(open(os.path.join(links_dir, "links.json"))) if os.path.exists(
        os.path.join(links_dir, "links.json")) else None
    robot = server.scene.add_frame("/g1", show_axes=False)
    static = server.scene.add_frame("/g1/static", show_axes=False)
    server.scene.add_glb("/g1/static/mesh", open(os.path.join(bundle, "robot.glb"), "rb").read())
    T_start_inv = np.linalg.inv(pose(*data["start"]))
    link_h, coll_h, hl_h = {}, {}, {}
    collision_frame = server.scene.add_frame("/g1/collision", show_axes=False)
    halo, halo_frame = {}, server.scene.add_frame("/hand_halo", show_axes=False)
    for s in SIDES:                                   # translucent halo + label around each robot hand
        f = server.scene.add_frame(f"/hand_halo/{s}", show_axes=False)
        server.scene.add_label(f"/hand_halo/{s}/label", f"{s[0].upper()} hand", position=(0, 0, .16))
        halo[s] = (f, {k: server.scene.add_icosphere(f"/hand_halo/{s}/{k}", radius=.13, color=c, opacity=.18,
                                                     cast_shadow=False, visible=False)
                       for k, c in STATE_COLORS.items()})

    def robot_hand_centres(tk):
        """Dex3 envelope centre of each robot hand over the take (cached on the take)."""
        if getattr(tk, "_hand_centres", None) is None:
            index = {n: k for k, n in enumerate(tk.g1["links"])}
            tk._hand_centres = {}
            for s in SIDES:
                g = rig["collision"][f"furniture_{s}_hand_envelope"]
                k = index[g["body"]]
                R = np.array([tf.SO3(q).as_matrix() for q in tk.g1["body_quat"][:, k]])
                tk._hand_centres[s] = tk.g1["body_pos"][:, k] + R @ np.asarray(g["pos"])
        return tk._hand_centres
    if rig:
        for name in rig["links"]:
            link_h[name] = server.scene.add_glb(f"/g1/links/{name}",
                                                open(os.path.join(links_dir, name + ".glb"), "rb").read(),
                                                visible=False)
        for name in rig.get("hand_links", []):            # Dex3 hand meshes in each clearance colour
            hl_h[name] = {k: server.scene.add_glb(f"/g1/hands/{name}_{k}",
                                                  open(os.path.join(links_dir, f"{name}__{k}.glb"), "rb").read(),
                                                  visible=False)
                          for k in STATE_COLORS}
        for gname, g in rig["collision"].items():     # robot collision geoms, one copy per state
            if g["type"] == "box":
                coll_h[gname] = {k: server.scene.add_box(f"/g1/collision/{gname}_{k}",
                                                         dimensions=tuple(2 * np.asarray(g["size"])), color=c,
                                                         opacity=.28, cast_shadow=False, visible=False)
                                 for k, c in STATE_COLORS.items()}
            else:
                r = g["size"][0]
                v, f = capsule_mesh(2 * (g["size"][1] + r) if g["type"] == "capsule" else 2 * r, r)
                coll_h[gname] = {k: server.scene.add_mesh_simple(f"/g1/collision/{gname}_{k}", v, f, color=c,
                                                                  opacity=.28, cast_shadow=False, visible=False)
                                 for k, c in STATE_COLORS.items()}

    # capture-time operator proxies (what the capture checked), one capsule mesh per state
    hv, hf = capsule_mesh(DEX3_ENVELOPE_END - DEX3_ENVELOPE_START, DEX3_ENVELOPE_RADIUS)
    fv, ff = capsule_mesh(FOREARM_LENGTH, FOREARM_RADIUS)
    tv, tf_ = capsule_mesh(G1_SHOULDER_HEIGHT - G1_PELVIS_HEIGHT - 0.1 + 2 * TORSO_RADIUS, TORSO_RADIUS)
    proxy_frame = server.scene.add_frame("/proxy", show_axes=False)
    proxies = {}
    for s in SIDES:
        for part, (v, f) in (("hand", (hv, hf)), ("forearm", (fv, ff))):
            proxies[s, part] = {k: server.scene.add_mesh_simple(f"/proxy/{s}_{part}_{k}", v, f, color=c,
                                                                 opacity=.45, cast_shadow=False, visible=False)
                                for k, c in STATE_COLORS.items()}
    proxies["body"] = {k: server.scene.add_mesh_simple(f"/proxy/body_{k}", tv, tf_, color=c, opacity=.15,
                                                        cast_shadow=False, visible=False)
                       for k, c in STATE_COLORS.items()}
    head_ball = server.scene.add_icosphere("/proxy/head", radius=HEAD_RADIUS, color=(60, 60, 60), opacity=.5)
    tracked_frame = server.scene.add_frame("/tracked", show_axes=False)
    body_frame = server.scene.add_frame("/body", show_axes=False)
    gaze = server.scene.add_frame("/tracked/gaze", axes_length=.25, axes_radius=.008)
    wrist_axes = {s: server.scene.add_frame(f"/tracked/{s}_wrist", axes_length=.07, axes_radius=.004) for s in SIDES}
    joints_pc = {s: server.scene.add_point_cloud(f"/tracked/{s}_joints", np.zeros((1, 3), np.float32),
                                                 colors=HAND_COLORS[s], point_size=.012, point_shape="circle")
                 for s in SIDES}

    # gui
    labels = [tk.label for tk in takes]
    with server.gui.add_folder("Take", expand_by_default=True):
        pick = server.gui.add_dropdown("Take", labels, initial_value=labels[0])
        auto = server.gui.add_checkbox("Auto-advance", initial_value=len(takes) > 1)
        info = server.gui.add_markdown("")
    with server.gui.add_folder("POV (headset view)", expand_by_default=True):
        pov_img = server.gui.add_image(np.full((360, 480, 3), 40, np.uint8), label="operator view", format="jpeg",
                                       jpeg_quality=80)
        pov_note = server.gui.add_markdown("")
    with server.gui.add_folder("Playback", expand_by_default=True):
        slider = server.gui.add_slider("Frame", min=0, max=max(takes[0].n - 1, 1), step=1, initial_value=0)
        playing = server.gui.add_checkbox("Play", initial_value=True)
        speed = server.gui.add_dropdown("Speed", ("0.25", "0.5", "1", "2"), initial_value="1")
        readout = server.gui.add_markdown("")
    with server.gui.add_folder("Show", expand_by_default=False):
        show_robot = server.gui.add_checkbox("G1", initial_value=True)
        show_hl = server.gui.add_checkbox("Highlight G1 hands (colour, halo, path)", initial_value=True)
        show_coll = server.gui.add_checkbox("G1 arm collision geoms (clearance colour)", initial_value=False)
        follow_cam = server.gui.add_checkbox("Camera follows the robot", initial_value=False)
        show_tracked = server.gui.add_checkbox("Tracked wrists + finger joints", initial_value=True)
        show_body = server.gui.add_checkbox("Tracked body (PICO skeleton, G1 scale)", initial_value=True)
        show_skel = server.gui.add_checkbox("G1 skeleton (robot links as lines)", initial_value=True)
        show_proxy = server.gui.add_checkbox("Capture-time hand proxies", initial_value=False)
        show_walls = server.gui.add_checkbox("Walls", initial_value=True)
        show_trails = server.gui.add_checkbox("Trails (current take)", initial_value=True)
        show_all = server.gui.add_checkbox("All takes (routes, head paths)", initial_value=True)
    proxy_frame.visible = False
    collision_frame.visible = False
    show_hl.on_update(lambda e: (setattr(halo_frame, "visible", e.target.value),
                                 setattr(current["hand_trails"], "visible", e.target.value)))
    for cb, handle in ((show_robot, robot), (show_coll, collision_frame), (show_tracked, tracked_frame), (show_body, body_frame),
                       (show_proxy, proxy_frame), (show_walls, walls), (show_all, overview)):
        cb.on_update(lambda e, handle=handle: setattr(handle, "visible", e.target.value))
    show_trails.on_update(lambda e: setattr(current["trails"], "visible", e.target.value))
    server.gui.add_markdown("G1 arms follow the tracked wrists by IK (7 DoF per arm); the base follows the head; "
                            "the feet follow the tracked ankles (native app takes) or planned steps. Orange "
                            "skeleton: the PICO body tracking at G1 scale. Highlighted hands: the "
                            "G1's Dex3 hands coloured by their clearance to the furniture (blue ok, orange within "
                            "the margin, red contact), with a halo and the path they took. Axes: tracked wrists; "
                            "dots: finger joints.")

    def show_state(handles, st, position=None, wxyz=None):
        for k, h in handles.items():
            if k == st:
                h.position, h.wxyz = position, wxyz
            h.visible = k == st

    def name(asset):
        return scene.labels.get(int(asset), "-") if asset >= 0 else "-"

    def quat_mul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

    lock = threading.RLock()
    cur = {"take": takes[0]}

    bones = [("pelvis", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("left_ankle", "left_foot"),
             ("pelvis", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
             ("right_ankle", "right_foot"), ("pelvis", "spine1"), ("spine1", "spine2"), ("spine2", "spine3"),
             ("spine3", "neck"), ("neck", "head"), ("spine3", "left_collar"), ("left_collar", "left_shoulder"),
             ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("left_wrist", "left_hand"),
             ("spine3", "right_collar"), ("right_collar", "right_shoulder"), ("right_shoulder", "right_elbow"),
             ("right_elbow", "right_wrist"), ("right_wrist", "right_hand")]

    skel_frame = server.scene.add_frame("/g1skel", show_axes=False)
    pov_state = {"k": -1, "take": None}

    def draw_skeleton(tk, i):
        g1 = tk.g1
        if g1 is None or "link_parent" not in g1 or not show_skel.value:
            skel_frame.visible = False
            return
        skel_frame.visible = True
        P, par = g1["body_pos"][i], g1["link_parent"]
        segs = np.array([[P[k], P[p]] for k, p in enumerate(par) if p >= 0], np.float32)
        if len(segs):
            server.scene.add_line_segments("/g1skel/bones", segs, colors=(20, 20, 20), thickness=.012)
            server.scene.add_point_cloud("/g1skel/joints", P.astype(np.float32), colors=(255, 255, 255), point_size=.03,
                                         point_shape="circle")

    def draw_pov(tk, i):
        if tk.pov_t is None:
            if pov_state["take"] is not tk:
                pov_img.image = np.full((360, 480, 3), 40, np.uint8)
                pov_note.content = "no POV video in this take (recorded before the app streamed it)"
                pov_state.update(take=tk, k=-1)
            return
        frames = tk.pov_frames()
        if not frames:
            return
        k = int(np.clip(np.searchsorted(tk.pov_t, tk.t[i]), 0, len(frames) - 1))
        if k != pov_state["k"] or pov_state["take"] is not tk:
            pov_img.image = frames[k]
            pov_note.content = f"frame {k + 1}/{len(frames)} · t {tk.pov_t[k]:.2f} s (tracking t {tk.t[i]:.2f} s)"
            pov_state.update(take=tk, k=k)

    def draw_body(tk, i):
        if tk.body is None:
            body_frame.visible = False
            return
        body_frame.visible = show_body.value
        idx = {n: k for k, n in enumerate(tk.body_names)}
        J = tk.body[i]
        segs, cols = [], []
        for a, b in bones:
            if a in idx and b in idx and np.all(np.isfinite(J[[idx[a], idx[b]]])):
                segs.append(J[[idx[a], idx[b]]])
                leg = any(w in a + b for w in ("hip", "knee", "ankle", "foot"))
                cols.append([(255, 140, 0)] * 2 if leg else [(34, 193, 230)] * 2)
        if segs:
            server.scene.add_line_segments("/body/bones", np.array(segs, np.float32), colors=np.array(cols, np.uint8),
                                           thickness=0.02)
            pts = J[np.all(np.isfinite(J), axis=1)].astype(np.float32)
            server.scene.add_point_cloud("/body/joints", pts, colors=(255, 140, 0), point_size=.025,
                                         point_shape="circle")

    def draw(i):
        tk = cur["take"]
        i = min(i, tk.n - 1)
        z, margin, g1 = tk.z, tk.margin, tk.g1
        head, yaw = tk.heads[i], tk.yaws[i]
        articulated = g1 is not None and rig is not None
        static.visible = not articulated
        for n, h in link_h.items():
            h.visible = articulated and not (n in hl_h and show_hl.value)
        if not articulated:
            for v in [v for d in hl_h.values() for v in d.values()] + [v for s in SIDES for v in halo[s][1].values()]:
                v.visible = False
        out = [f"t = {tk.t[i]:.2f} s · frame {i}/{tk.n - 1}"]
        if articulated:
            index = {n: k for k, n in enumerate(g1["links"])}
            hand_state = {s: state(g1[f"clear_{s}_hand"][i], margin) for s in SIDES}
            for n, h in link_h.items():
                k = index[n]
                p, qb = tuple(g1["body_pos"][i, k]), tuple(g1["body_quat"][i, k])
                if n in hl_h and show_hl.value:
                    h.visible = False
                    show_state(hl_h[n], hand_state["left" if n.startswith("left") else "right"], p, qb)
                else:
                    h.position, h.wxyz = p, qb
                    for v in hl_h.get(n, {}).values():
                        v.visible = False
            centres = robot_hand_centres(tk)
            for s in SIDES:
                halo[s][0].position = tuple(centres[s][i])
                show_state(halo[s][1], hand_state[s], (0.0, 0.0, 0.0), (1.0, 0, 0, 0))
            if follow_cam.value:
                target = np.r_[g1["qpos"][i, :2], 0.75]
                for client in server.get_clients().values():
                    off = np.asarray(client.camera.position) - np.asarray(client.camera.look_at)
                    with client.atomic():
                        client.camera.look_at = tuple(target)
                        client.camera.position = tuple(target + off)
            for gname, g in rig["collision"].items():
                k = index[g["body"]]
                p, qb = g1["body_pos"][i, k], g1["body_quat"][i, k]
                R = tf.SO3(qb).as_matrix()
                show_state(coll_h[gname], state(g1[f"clear_{g['group']}"][i], margin),
                           tuple(p + R @ np.asarray(g["pos"])), tuple(quat_mul(qb, g["quat"])))
            fn = g1["furniture_names"]
            nm = lambda k: str(fn[k]).removeprefix("furniture_") if k >= 0 else "-"
            out.append("**G1 clearance** (robot geometry)")
            for s in SIDES:
                out.append(f"{s}: hand {g1[f'clear_{s}_hand'][i]:+.3f} · forearm {g1[f'clear_{s}_forearm'][i]:+.3f}"
                           f" · upper arm {g1[f'clear_{s}_upper_arm'][i]:+.3f} m ({nm(g1[f'near_{s}_hand'][i])})")
            out.append(f"body {g1['clear_body'][i]:+.3f} m")
            errs = [f"{s} {g1[f'{s}_pos_err'][i] * 100:.1f} cm" for s in SIDES if np.isfinite(g1[f"{s}_pos_err"][i])]
            out.append("wrist IK error: " + (", ".join(errs) or "-") + "  (>3 cm: beyond the G1's reach)")
        else:
            T = pose(head[0], head[1], yaw) @ T_start_inv
            robot.position = (T[0, 3], T[1, 3], 0.0)
            robot.wxyz = tf.SO3.from_z_radians(np.arctan2(T[1, 0], T[0, 0])).wxyz
            out.append("no arm retarget for this take: run `view_take prepare`")
        draw_body(tk, i)
        draw_skeleton(tk, i)
        draw_pov(tk, i)
        bad = [s for s in SIDES if s in tk.glitch and tk.glitch[s][i]]
        if bad:
            out.append("**CONTROLLER GLITCH: " + ", ".join(bad) + "** (tracking lost; ignored for clearance)")
        if articulated and "feet_source" in g1:
            out.append(f"feet: {g1['feet_source']}"
                       + (f" (IK error L {g1['left_foot_err'][i] * 100:.1f} / R {g1['right_foot_err'][i] * 100:.1f} cm)"
                          if str(g1["feet_source"]) == "tracked" else ""))
        head_ball.position = tuple(head)
        gaze.position, gaze.wxyz = tuple(head), tf.SO3.from_z_radians(yaw).wxyz
        top = head - [0, 0, G1_EYE_HEIGHT - G1_SHOULDER_HEIGHT]
        bottom = head - [0, 0, G1_EYE_HEIGHT - G1_PELVIS_HEIGHT - 0.1]
        show_state(proxies["body"], state(z["clear_body"][i], margin), tuple((top + bottom) / 2), (1.0, 0, 0, 0))
        out.append("**capture proxy** (operator, G1 scale)")
        for s in SIDES:
            h = tk.hands[s]
            ok = bool(h["tracked"][i])
            wrist_axes[s].visible = joints_pc[s].visible = ok
            if not ok:
                for part in ("hand", "forearm"):
                    show_state(proxies[s, part], None)
                out.append(f"{s}: not tracked (arm holds its last pose)")
                continue
            w, R = h["wrist"][i], h["R"][i]
            fwd = R[:, 0] if tk.ctrl else -R[:, 2]        # wrist -> fingers
            rot = tuple(wxyz_from_matrix(z_to(fwd)))
            show_state(proxies[s, "hand"], state(h["clear_hand"][i], margin),
                       tuple(w + fwd * (DEX3_ENVELOPE_START + DEX3_ENVELOPE_END) / 2), rot)
            show_state(proxies[s, "forearm"], state(h["clear_forearm"][i], margin),
                       tuple(w - fwd * FOREARM_LENGTH / 2), rot)
            wrist_axes[s].position, wrist_axes[s].wxyz = tuple(w), tuple(wxyz_from_matrix(R))
            if h["joints"] is not None and np.all(np.isfinite(h["joints"][i])):
                joints_pc[s].points = h["joints"][i].astype(np.float32)
            out.append(f"{s}: hand {h['clear_hand'][i]:+.3f} · forearm {h['clear_forearm'][i]:+.3f} m "
                       f"({name(h['near_hand'][i])})")
        readout.content = "  \n".join(out)

    def select(k):
        with lock:
            tk = takes[k]
            cur["take"] = tk
            build_current(tk)
            info.content = tk.summary()
            slider.max = max(tk.n - 1, 1)
            slider.value = 0
            draw(0)

    pick.on_update(lambda e: select(labels.index(pick.value)))
    @slider.on_update
    def _(_):
        with lock:
            draw(int(slider.value))

    # camera: over the whole set of routes, from behind the first segment start
    pts = np.vstack([np.asarray(tk.seg["route"], float) for tk in takes])
    mid, span = pts.mean(0), max(np.ptp(pts, axis=0).max(), 2.0)
    sx, sy, syaw = takes[0].seg["start"]
    fx, fy = np.cos(syaw), np.sin(syaw)
    server.initial_camera.position = (mid[0] - .9 * span * fx + .5 * span * fy, mid[1] - .9 * span * fy - .5 * span * fx,
                                      .9 * span + 1)
    server.initial_camera.look_at = (mid[0], mid[1], .6)
    server.initial_camera.up_direction = (0, 0, 1)

    def follow(client):
        tk = cur["take"]
        i = min(int(slider.value), tk.n - 1)
        h, yaw = tk.heads[i], tk.yaws[i]
        c, s = np.cos(yaw), np.sin(yaw)
        return (h[0] - 1.6 * c + .7 * s, h[1] - 1.6 * s - .7 * c, 1.9), (h[0] + .8 * c, h[1] + .8 * s, .8)
    views = {
        "Overview": lambda c: (server.initial_camera.position, server.initial_camera.look_at),
        "Behind the operator (now)": follow,
        "Top": lambda c: ((mid[0], mid[1], 1.6 * span + 2), (mid[0] + 1e-3, mid[1], 0.0)),
    }
    with server.gui.add_folder("Cameras", expand_by_default=False):
        for label, fn in views.items():
            b = server.gui.add_button(label)

            @b.on_click
            def _(event, fn=fn):
                if event.client is not None:
                    p, target = fn(event.client)
                    with event.client.atomic():
                        event.client.camera.position, event.client.camera.look_at = p, target
                        event.client.camera.up_direction = (0, 0, 1)

    select(0)
    print(f"http://127.0.0.1:{server.get_port()}   {data['scene_id']}: {len(takes)} take(s)", flush=True)
    return server, takes, cur, slider, playing, speed, auto, pick, labels


def run(groups, port):
    servers = [serve(paths, port + k) for k, paths in enumerate(groups)]
    last = time.monotonic()
    try:
        while True:
            time.sleep(0.01)
            now = time.monotonic()
            for server, takes, cur, slider, playing, speed, auto, pick, labels in servers:
                tk = cur["take"]
                dt = float(np.median(np.diff(tk.t))) if tk.n > 1 else 0.05
                if not playing.value or now - cur.get("last", 0) < dt / float(speed.value):
                    continue
                cur["last"] = now
                i = int(slider.value) + 1
                if i < tk.n:
                    slider.value = i
                elif auto.value and len(takes) > 1:
                    pick.value = labels[(takes.index(tk) + 1) % len(takes)]
                else:
                    slider.value = 0
    except KeyboardInterrupt:
        for s in servers:
            s[0].stop()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["prepare", "serve"])
    p.add_argument("takes", nargs="*", help="take directories")
    p.add_argument("--successful", action="store_true", help="every take with status success and safe")
    p.add_argument("--takes-dir", default=TAKES)
    p.add_argument("--port", type=int, default=8110, help="first port (one per scene)")
    a = p.parse_args()
    takes = list(a.takes) + (successful_takes(a.takes_dir) if a.successful else [])
    if not takes:
        raise SystemExit("no takes (pass take directories or --successful)")
    if a.command == "prepare":
        prepare(takes)
        return
    groups = {}
    for t in takes:
        groups.setdefault(scene_json_of(t, load_meta(t)), []).append(t)
    run(list(groups.values()), a.port)


if __name__ == "__main__":
    main()
