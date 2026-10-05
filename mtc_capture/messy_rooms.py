"""Messy rooms for hand-protection capture, certified with the real G1 collision geometry.

Rooms come from chaotic_rooms.generate_chaotic_room (cluttered tables, tossed and toppled chairs,
crate towers, planks, poles, shelves, open cabinets, partitions, with hand-height things lining
the routes). A route is kept only if the G1 (29 DoF + Dex3, the branch's collision geoms in
MuJoCo, standing upright at every point of the route, facing along it) shows that it is

  passable    with the arms held in (hands in front of the chest), no hand, forearm, upper arm
              or torso geom comes within CLEAR_MARGIN of any object, also with the base shifted
              sideways by +-ROBUST_SHIFT and turned by +-ROBUST_YAW;
  hand-gated  with a normal walking arm swing (arms hanging, shoulders swinging +-SWING), the arms
              hit at least MIN_SWING_HITS different objects: walking through without moving the
              hands is not possible;
  winding     the route turns at least MIN_TURN in total and bends away from the straight start
              -> goal line by at least MIN_DETOUR: the way leads around things;
  walkable    every capture segment fits the real floor (FLOOR, at the tallest operator's scale).

A room is kept if at least one route passes; failing routes are dropped from scene["routes"]
(start_goals keep their indices). The certificate is stored in scene["g1_certificate"].

    source .venv/bin/activate && source .env
    python -m mtc_capture.messy_rooms --name messy_v1 --rooms 4      # -> data/mtc_capture/scene_sets/messy_v1
"""

import argparse
import json
import math
import os
import shutil
import tempfile

import numpy as np

from .arm_ik import GROUPS, BODY_GEOMS, SIDES, _ik, build_model

CLEAR_MARGIN = 0.03          # m: arms-in pose must clear everything by this much
ROBUST_SHIFT = 0.03          # m: ... also with the base this far to either side
ROBUST_YAW = math.radians(8)
SWING = math.radians(25)     # shoulder pitch amplitude of a normal walking arm swing
MIN_SWING_HITS = 2           # distinct objects the swinging arms hit
MIN_TURN = math.radians(60)  # total heading change along the route
MIN_DETOUR = 0.30            # m: max distance of the route from its start -> goal line
FLOOR = (5.0, 3.0)           # m: real free floor (forward, width)
TALLEST = 1.84               # m: tallest operator -> smallest alpha -> most floor needed
SPACING = 0.05               # m between checked poses along the route
PATTERNS = ("hard_dense", "office_dense", "storage_dense", "debris_dense", "classroom")


def _register_dense_patterns():
    """Denser variants of chaotic_rooms patterns: one certified route per room is enough, so the
    generator does not thin the clutter to make room for three routes; 30 % more objects and
    hand-height clutter lining both sides of the route."""
    from . import chaotic_rooms as CR
    for base in ("hard", "office", "storage", "debris"):
        p = dict(CR.PATTERNS[base])
        lo, hi = p["count"]
        p.update(min_routes=1, count=(round(1.3 * lo), round(1.3 * hi)), both_sides=True,
                 side_spacing=min(p["side_spacing"], .7))
        CR.PATTERNS[f"{base}_dense"] = p
        CR.TARGET[f"{base}_dense"] = p["target"]
        # tight: a smaller room with more in it, so the clutter crowds the route
        t = dict(p, count=(round(1.2 * lo), round(1.2 * hi)),
                 dims=tuple((round(.8 * a, 2), round(.8 * b, 2)) for a, b in CR.PATTERNS[base]["dims"]))
        CR.PATTERNS[f"{base}_tight"] = t
        CR.TARGET[f"{base}_tight"] = t["target"]
    CR.DIFFICULTIES = tuple(CR.PATTERNS)


_register_dense_patterns()


# ----------------------------------------------------------------- robot poses

class G1Checker:
    """The G1 placed along a route in one scene's MuJoCo model, with signed distances of its arm
    and torso collision geoms to the furniture."""

    def __init__(self, scene_dir):
        import mujoco
        self.mj = mujoco
        self.model, self.q0 = build_model(scene_dir)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.furniture = np.flatnonzero((m.geom_bodyid == 0) & (m.geom_contype == 2))
        self.fnames = [m.geom(g).name for g in self.furniture]
        self.groups = {f"{s}_{k}": [m.geom(x.format(s=s)).id for x in (v if isinstance(v, tuple) else (v,))]
                       for s in SIDES for k, v in GROUPS.items()}
        self.groups["body"] = [m.geom(x).id for x in BODY_GEOMS]
        self.arm_q = {s: [m.jnt_qposadr[m.joint(f"{s}_{j}_joint").id] for j in
                          ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch",
                           "wrist_yaw")] for s in SIDES}
        self.poses = self._poses()

    def _poses(self):
        """Joint vectors (base at the origin, facing +x): arms in, and the walking swing extremes."""
        m, d, mj = self.model, self.data, self.mj
        q = self.q0.copy()
        q[0:3], q[3:7] = [0, 0, self.q0[2]], [1, 0, 0, 0]
        d.qpos[:] = q
        mj.mj_kinematics(m, d)
        torso = d.xpos[m.body("torso_link").id]
        tucked = q.copy()
        J = np.zeros((6, m.nv))
        for s, sign in (("left", 1.0), ("right", -1.0)):   # wrists in front of the chest, close to the body
            jid = [m.joint(f"{s}_{j}_joint").id for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                                          "wrist_roll", "wrist_pitch", "wrist_yaw")]
            target = torso + np.array([0.22, sign * 0.09, 0.05])
            sol, cost = _ik(m, d, tucked, m.jnt_qposadr[jid], m.jnt_dofadr[jid], m.jnt_range[jid],
                            m.body(f"{s}_wrist_yaw_link").id, target, np.eye(3), np.r_[np.ones(3), np.full(3, 1e-3)],
                            200, 0.03, J)
            tucked[m.jnt_qposadr[jid]] = sol
        poses = {"tucked": tucked}
        for k, a in (("swing_fwd", -SWING), ("swing_back", SWING)):   # left and right swing opposite
            p = q.copy()
            p[self.arm_q["left"][0]] += a
            p[self.arm_q["right"][0]] -= a
            poses[k] = p
        poses["hanging"] = q.copy()
        return poses

    def clearance(self, pose, x, y, yaw, cap=0.5):
        """Signed distance per group and the nearest furniture geom, robot at (x, y, yaw)."""
        m, d, mj = self.model, self.data, self.mj
        q = self.poses[pose].copy()
        q[0], q[1] = x, y
        q[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qpos[:] = q
        mj.mj_kinematics(m, d)
        fromto = np.zeros(6)
        fpos, frad = d.geom_xpos[self.furniture], np.linalg.norm(m.geom_aabb[self.furniture, 3:], axis=1)
        out = {}
        for g, geoms in self.groups.items():
            best, who = cap, -1
            for gid in geoms:
                d0 = np.linalg.norm(fpos - d.geom_xpos[gid], axis=1) - frad - m.geom_rbound[gid]
                for k in np.flatnonzero(d0 < best):
                    dist = mj.mj_geomDistance(m, d, gid, self.furniture[k], best, fromto)
                    if dist < best:
                        best, who = dist, k
            out[g] = (best, who)
        return out


# ----------------------------------------------------------------- route geometry

def route_samples(route, spacing=SPACING):
    """(x, y, yaw) every spacing m; at corners both the incoming and the outgoing heading."""
    out = []
    for a, b in zip(route, route[1:]):
        L = math.dist(a, b)
        if L < 1e-6:
            continue
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        n = max(1, math.ceil(L / spacing))
        out += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n, yaw) for k in range(n + 1)]
    return out


def winding(route):
    turn = 0.0
    for a, b, c in zip(route, route[1:], route[2:]):
        h1, h2 = math.atan2(b[1] - a[1], b[0] - a[0]), math.atan2(c[1] - b[1], c[0] - b[0])
        turn += abs(math.remainder(h2 - h1, 2 * math.pi))
    p0, p1 = np.asarray(route[0][:2]), np.asarray(route[-1][:2])
    u = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
    detour = max(abs(float(np.cross(u, np.asarray(p[:2]) - p0))) for p in route)
    length = sum(math.dist(a[:2], b[:2]) for a, b in zip(route, route[1:]))
    return dict(total_turn_deg=round(math.degrees(turn), 1), max_detour_m=round(detour, 3),
                length_m=round(length, 2), straight_m=round(float(np.linalg.norm(p1 - p0)), 2))


def object_of(fname, scene):
    """Furniture geom 'furniture_object_<k>' (scene box k) -> the object it belongs to, None for walls."""
    k = int(fname.rsplit("_", 1)[1])
    b = scene["boxes"][k]
    if b.get("category") == "wall":
        return None
    return b.get("object_id") or b.get("furniture_id") or b["name"]


def certify_route(chk, scene, route):
    tucked_min, tucked_at = 1.0, None
    swing_hits = {}
    for x, y, yaw in route_samples(route):
        for dx in (0.0, ROBUST_SHIFT, -ROBUST_SHIFT):
            for dyaw in (0.0, ROBUST_YAW, -ROBUST_YAW):
                ox, oy = -math.sin(yaw) * dx, math.cos(yaw) * dx
                c = chk.clearance("tucked", x + ox, y + oy, yaw + dyaw)
                for g, (dist, who) in c.items():
                    if dist < tucked_min:
                        tucked_min, tucked_at = dist, (g, chk.fnames[who] if who >= 0 else "-", round(x, 2), round(y, 2))
        for pose in ("swing_fwd", "swing_back", "hanging"):
            c = chk.clearance(pose, x, y, yaw)
            for g, (dist, who) in c.items():
                if g != "body" and dist < 0 and who >= 0:
                    obj = object_of(chk.fnames[who], scene)
                    if obj is not None:
                        swing_hits.setdefault(obj, []).append(g)
    w = winding(route)
    res = dict(tucked_min_clearance_m=round(tucked_min, 3), tucked_nearest=tucked_at,
               swing_hit_objects=sorted(swing_hits), swing_hits=len(swing_hits), **w)
    res["passable"] = tucked_min >= CLEAR_MARGIN
    res["hand_gated"] = len(swing_hits) >= MIN_SWING_HITS
    res["winding"] = math.radians(w["total_turn_deg"]) >= MIN_TURN and w["max_detour_m"] >= MIN_DETOUR
    return res


def floor_fit(scene_dict, case_index, segment_length=3.0):
    from .furniture import FurnitureScene
    sc = FurnitureScene(scene_dict, "messy")
    alpha = 1.32 / TALLEST
    view = sc.route_views()[[r["case_index"] for r in scene_dict["routes"]].index(case_index)]
    segs = view.segments(segment_length, "auto", (alpha, *FLOOR))
    need = [view.floor_need(s.s0, s.s1, alpha) for s in segs]
    ok = all(f <= FLOOR[0] + .05 and w <= FLOOR[1] + .05 for f, w in need)
    return ok, len(segs), [tuple(round(v, 2) for v in n) for n in need]


# ----------------------------------------------------------------- set builder

def certify_room(scene):
    """-> (scene with only the certified routes, report) or (None, report)."""
    tmp = tempfile.mkdtemp(prefix="messy_")
    try:
        with open(os.path.join(tmp, "scene.json"), "w") as f:
            json.dump(scene, f)
        chk = G1Checker(tmp)
        cases = scene["start_goals"]
        kept, report = [], []
        for r in scene["routes"]:
            res = certify_route(chk, scene, cases[r["case_index"]]["route"])
            res["route"] = r["name"]
            if res["passable"] and res["hand_gated"] and res["winding"]:
                fit, nseg, need = floor_fit(scene, r["case_index"])
                res.update(floor_fit=fit, segments=nseg, floor_need_m=need)
                if fit:
                    kept.append(dict(r, g1_certificate=res))
            report.append(res)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not kept:
        return None, report
    out = dict(scene, routes=kept)
    first = kept[0]["case_index"]   # the room's main route = its best certified one
    out.update(goal_index=first, **{k: cases[first][k] for k in cases[first]})
    out["g1_certificate"] = dict(
        method="G1 29-DoF + Dex3 collision geoms (MuJoCo, the branch's assemble_scene_xml) upright along the route, "
               "facing along it", clear_margin_m=CLEAR_MARGIN, robust_shift_m=ROBUST_SHIFT,
        robust_yaw_deg=math.degrees(ROBUST_YAW), swing_deg=math.degrees(SWING), min_swing_hits=MIN_SWING_HITS,
        min_turn_deg=math.degrees(MIN_TURN), min_detour_m=MIN_DETOUR, floor_m=FLOOR, tallest_operator_m=TALLEST,
        routes=[r["g1_certificate"] for r in kept])
    return out, report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="messy_v1")
    p.add_argument("--rooms", type=int, default=4, help="rooms to keep (one per pattern, in order)")
    p.add_argument("--patterns", nargs="+", default=list(PATTERNS))
    p.add_argument("--max-seeds", type=int, default=12, help="seeds tried per pattern")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "mtc_capture", "scene_sets"))
    a = p.parse_args()
    from .chaotic_rooms import generate_chaotic_room
    from .furniture import FurnitureScene
    from .make_scene_set import overview

    root = os.path.abspath(os.path.join(a.out, a.name))
    entries = []
    for pattern in a.patterns:
        if len(entries) >= a.rooms:
            break
        for seed in range(a.max_seeds):
            try:
                scene = generate_chaotic_room(seed, "train", pattern)
            except RuntimeError as e:   # no walkable layout for this seed (too crowded)
                print(f"  {pattern:9s} seed {seed:2d} {e}")
                continue
            cert, report = certify_room(scene)
            for r in report:
                print(f"  {pattern:9s} seed {seed:2d} route {r['route']:10s} tucked {r['tucked_min_clearance_m']:+.3f} m "
                      f"swing hits {r['swing_hits']} turn {r['total_turn_deg']:5.0f} deg detour {r['max_detour_m']:.2f} m "
                      f"{'PASS' if r['passable'] and r['hand_gated'] and r['winding'] and r.get('floor_fit') else ''}"
                      + (f" segments {r.get('segments')} floor {r.get('floor_need_m')}" if 'segments' in r else ""))
            if cert is None:
                continue
            sc = FurnitureScene(cert, f"chaotic:{pattern}")
            rel = os.path.join("scenes", "train", "chaotic", sc.scene_id, "scene.json")
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            with open(os.path.join(root, rel), "w") as f:
                json.dump(cert, f)
            segs = sc.all_segments(3.0, "auto", (1.32 / TALLEST, *FLOOR))
            entries.append(dict(path=rel, scene_id=sc.scene_id, split="train", family="chaotic", variant=pattern,
                                role=None, seed=seed, route_length_m=round(sc.route_length, 2), hazards=0,
                                routes=len(cert["routes"]), segments=len(segs), takes_per_segment=3,
                                planned_takes=3 * len(segs), root_route_validated=True, hand_contrast_targets=False,
                                g1_certificate=cert["g1_certificate"]["routes"]))
            print(f"KEPT {sc.scene_id}: {len(cert['routes'])} certified route(s), {len(segs)} segments")
            break
    manifest = dict(schema="mtc-capture-scene-set-v1", name=a.name, segment_length_m=3.0,
                    takes_per_segment=dict(train=3, validation=1, test=0),
                    plan=dict(messy=dict(patterns=a.patterns, rooms=a.rooms, certificate=__doc__.split("\n\n")[1])),
                    summary=dict(train=dict(scenes=len(entries), segments=sum(e["segments"] for e in entries),
                                            planned_takes=sum(e["planned_takes"] for e in entries))),
                    capture_order=[e["path"] for e in entries], scenes=entries)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    overview(root, entries)
    print(os.path.join(root, "manifest.json"), f"{len(entries)} rooms")


if __name__ == "__main__":
    main()
