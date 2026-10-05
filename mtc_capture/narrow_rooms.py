"""Narrow chaotic rooms: junk everywhere, and pinches along the route that the G1 only passes by
pulling its arms in (or raising its hands) or by turning sideways.

A base room comes from chaotic_rooms (a "junk" mix: crate towers, shelves, open cabinets,
partitions, planks, poles, tossed and toppled chairs, at most one table). Its route keeps the
branch's 0.23 m root clearance, which a G1 walks through frontally with its arms in. So pinches
are added: about every metre, two chaotic objects face each other across the route, slid until the
gap between them is
    arms   0.46-0.60 m (G1 scale): too narrow to walk with arms swinging, wide enough with the
           arms held in (hands in front of the chest) or raised
    side   0.33-0.38 m: too narrow for the G1 facing forward in any arm pose; it passes sideways
Objects a pinch overlaps are removed. Every 5 cm of the route is then classified with the real
G1 collision geometry (messy_rooms.G1Checker, legs included), robust to a 2 cm sideways shift and
a 5 degree turn:
    free      walking normally (arms hanging or swinging) clears everything
    arms      only with the arms in or the hands raised
    side      only sideways (arms hanging or raised)
    blocked   none of these: the pinch is rejected
A pinch is kept only if its stretch needs exactly the strategy it was built for. A room is kept if
nothing on the route is blocked, it has at least MIN_SIDE sideways and MIN_PINCHES pinches in
all, the route still winds, and every capture segment fits the real floor.

Measured minimum gaps (G1 scale, full-height objects): arms swinging 0.68-0.70 m, arms in 0.40 m,
hands raised 0.58 m (0.40 m over hand-height clutter), sideways 0.27 m. For the G1, holding the
arms in passes everywhere raising the hands does, so no pinch can require raised hands; they are
accepted as an alternative.

    source .venv/bin/activate && source .env
    python -m mtc_capture.narrow_rooms --name messy_v2 --rooms 4
"""

import argparse
import copy
import json
import math
import os
import random
import shutil
import tempfile

import numpy as np

from . import messy_rooms as MR

MARGIN = 0.04                 # m: clearance a strategy needs (with the shifts below)
SHIFT = 0.03                  # m: sideways shift of the base for robustness
TURN = math.radians(6)
GAPS = {"arms": (0.56, 0.70), "side": (0.42, 0.47)}   # v2 (too hard to do cleanly): 0.46-0.60 / 0.33-0.38
MIN_SIDE = 1
MIN_PINCHES = 3
PINCH_HALF = 0.3              # m: route stretch on each side of a pinch that must need its strategy
END_KEEP = 0.8                # m: no pinch this close to the start or goal pad
GENERATOR = "mtc-narrow-chaotic-v1"


# ----------------------------------------------------------------- strategies

class Strategies(MR.G1Checker):
    """G1Checker with the legs, a hands-raised pose and the sideways strategies."""

    def __init__(self, scene_dir):
        super().__init__(scene_dir)
        m = self.model
        self.robot = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and m.geom_contype[g] != 0]
        self.poses["raised"] = self._raised()

    def _raised(self):
        m, d, mj = self.model, self.data, self.mj
        q = self.poses["hanging"].copy()
        q[0:3], q[3:7] = [0, 0, self.q0[2]], [1, 0, 0, 0]
        d.qpos[:] = q
        mj.mj_kinematics(m, d)
        torso = d.xpos[m.body("torso_link").id].copy()
        J = np.zeros((6, m.nv))
        for s, sign in (("left", 1.0), ("right", -1.0)):   # wrists above the head
            jid = [m.joint(f"{s}_{j}_joint").id for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                                          "wrist_roll", "wrist_pitch", "wrist_yaw")]
            sol, _ = MR._ik(m, d, q, m.jnt_qposadr[jid], m.jnt_dofadr[jid], m.jnt_range[jid],
                            m.body(f"{s}_wrist_yaw_link").id, torso + np.array([0.05, sign * 0.13, 0.42]), np.eye(3),
                            np.r_[np.ones(3), np.full(3, 1e-3)], 300, 0.03, J)
            q[m.jnt_qposadr[jid]] = sol
        return q

    def min_clear(self, pose, x, y, yaw, cap=0.3):
        """Smallest signed distance of any robot collision geom (arms, torso, head, pelvis, legs) to the furniture."""
        m, d, mj = self.model, self.data, self.mj
        q = self.poses[pose].copy()
        q[0], q[1] = x, y
        q[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qpos[:] = q
        mj.mj_kinematics(m, d)
        fpos, frad = d.geom_xpos[self.furniture], np.linalg.norm(m.geom_aabb[self.furniture, 3:], axis=1)
        best, fromto = cap, np.zeros(6)
        for g in self.robot:
            d0 = np.linalg.norm(fpos - d.geom_xpos[g], axis=1) - frad - m.geom_rbound[g]
            for k in np.flatnonzero(d0 < best):
                best = min(best, mj.mj_geomDistance(m, d, g, self.furniture[k], best, fromto))
        return best

    def robust(self, pose, x, y, yaw):
        nx, ny = -math.sin(yaw), math.cos(yaw)
        return min(self.min_clear(pose, x + nx * s, y + ny * s, yaw + t)
                   for s in (0.0, SHIFT, -SHIFT) for t in (0.0, TURN, -TURN))

    def label(self, x, y, yaw):
        if min(self.min_clear(p, x, y, yaw) for p in ("hanging", "swing_fwd", "swing_back")) >= 0.0:
            return "free"
        if self.robust("tucked", x, y, yaw) >= MARGIN or self.robust("raised", x, y, yaw) >= MARGIN:
            return "arms"
        for side in (math.pi / 2, -math.pi / 2):
            if self.robust("hanging", x, y, yaw + side) >= MARGIN or self.robust("raised", x, y, yaw + side) >= MARGIN:
                return "side"
        return "blocked"


def labels_along(chk, samples):
    return [chk.label(x, y, yaw) for x, y, yaw in samples]


def runs(labels, s_of):
    """Contiguous stretches of non-free labels: [(label, s0, s1)] (side wins inside a stretch)."""
    out, cur = [], None
    for lab, s in zip(labels, s_of):
        if lab == "free":
            cur = None
            continue
        if cur is None:
            cur = [lab, s, s]
            out.append(cur)
        cur[2] = s
        if lab in ("side", "blocked") and cur[0] == "arms":
            cur[0] = lab
    return [tuple(r) for r in out]


# ----------------------------------------------------------------- geometry helpers

def polyline(route):
    pts = np.asarray([p[:2] for p in route], float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return pts, np.r_[0.0, np.cumsum(seg)]


def at(route, s):
    pts, cum = polyline(route)
    k = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(pts) - 2))
    t = pts[k + 1] - pts[k]
    L = np.linalg.norm(t)
    u = t / L
    return pts[k] + u * (s - cum[k]), u


def samples_with_s(route, spacing=MR.SPACING):
    pts, cum = polyline(route)
    out, ss = [], []
    for k in range(len(pts) - 1):
        L = cum[k + 1] - cum[k]
        if L < 1e-6:
            continue
        yaw = math.atan2(*(pts[k + 1] - pts[k])[::-1])
        n = max(1, math.ceil(L / spacing))
        for j in range(n + 1):
            out.append((*(pts[k] + (pts[k + 1] - pts[k]) * j / n), yaw))
            ss.append(cum[k] + L * j / n)
    return out, np.array(ss)


def box_dist(p, boxes, z=(0.05, 1.45)):
    from cat_ppo.furniture.scenes import _horizontal_distance
    ds = [_horizontal_distance(p, b) for b in boxes
          if b["center"][2] - b["half_size"][2] < z[1] and b["center"][2] + b["half_size"][2] > z[0]]
    return min(ds) if ds else 9.0


def aabb(boxes):
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for b in boxes:
        c, h, yaw = np.asarray(b["center"][:2]), np.asarray(b["half_size"][:2]), b.get("yaw", 0.0)
        e = np.array([abs(math.cos(yaw)) * h[0] + abs(math.sin(yaw)) * h[1],
                      abs(math.sin(yaw)) * h[0] + abs(math.cos(yaw)) * h[1]])
        lo, hi = np.minimum(lo, c - e), np.maximum(hi, c + e)
    return lo, hi


# ----------------------------------------------------------------- pinch objects

def junk_tower(rng, name, split, index):
    """Boxes stacked crooked, 0.9-1.5 m: a tall thing to squeeze past."""
    from cat_ppo.furniture.scenes import _box
    from .chaotic_rooms import _common
    c = _common(name, "junk_tower", split, index)
    z, parts = 0.0, []
    for k in range(rng.randint(3, 5)):
        h = rng.uniform(.18, .35)
        hx, hy = rng.uniform(.14, .26), rng.uniform(.12, .22)
        parts.append(_box(f"{name}_b{k}", [rng.uniform(-.06, .06), rng.uniform(-.06, .06), z + h / 2], [hx, hy, h / 2],
                          "crate", rng.uniform(-.4, .4), **c))
        z += h
    return parts


def pinch_builders(kind):
    from . import chaotic_rooms as CR
    if kind == "side":   # tall: the whole body has to fit
        return [junk_tower, CR.crate_stack, CR.crate_stack, CR.shelf_with_stuff, CR.open_cabinet, CR.open_cabinet]
    return [CR.messy_table, CR.messy_table, CR.route_side_hazard, CR.crate_stack, junk_tower, CR.pole_on_chair]


def place_object(rng, parts, route, s, side_sign, gap_half, others_window, keep_route, dims, tall):
    """Rotate/translate parts beside route point s so their nearest face in the body band is
    gap_half from the route (within the pinch stretch). -> placed parts or None."""
    from cat_ppo.furniture.random_rooms import transform_object
    p, u = at(route, s)
    n = np.array([-u[1], u[0]]) * side_sign
    yaw = math.atan2(u[1], u[0]) + rng.uniform(-.6, .6) + (math.pi if rng.random() < .5 else 0.0)
    band = (0.05, 1.45) if tall else (0.45, 1.25)
    if tall:   # sideways pinches need something tall enough that the hips and torso count
        top = max(b["center"][2] + b["half_size"][2] for b in parts)
        if top < 0.95:
            return None
    lo, hi = aabb(parts)
    off = gap_half + 0.5 * float(np.linalg.norm(hi - lo)) * 0.5
    for _ in range(8):
        c = p + n * off + u * rng.uniform(-.05, .05)
        placed = transform_object(parts, float(c[0]), float(c[1]), yaw)
        d = min(box_dist(q, placed, band) for q in others_window)
        if abs(d - gap_half) < 0.004:
            break
        off += gap_half - d
    else:
        return None
    lo, hi = aabb(placed)
    if lo[0] < .12 or lo[1] < .12 or hi[0] > dims[0] - .12 or hi[1] > dims[1] - .12:
        return None
    if min(box_dist(q, placed) for q in keep_route) < 0.26:   # the rest of the route keeps its clearance
        return None
    return placed


# ----------------------------------------------------------------- room

def register_junk():
    from . import chaotic_rooms as CR
    if "junk" in CR.PATTERNS:
        return
    p = dict(CR.PATTERNS["hard"])
    p.update(weights=dict(crate_stack=4, shelf=2, open_cabinet=2, partition=2, plank_bridge=2, pole_on_chair=2,
                          toppled_chair=2, tossed_chair=3, floor_lamp=1, floor_junk=2, pallet=1, plank_pile=2,
                          messy_table=4),
             count=(17, 23), dims=((5.4, 6.4), (4.8, 5.8)), min_routes=1, both_sides=True, side_spacing=.8)
    CR.PATTERNS["junk"] = p
    CR.TARGET["junk"] = p["target"]
    CR.DIFFICULTIES = tuple(CR.PATTERNS)


def build_scene(base, boxes, route, profile):
    from cat_ppo.furniture.scenes import _case, _digest, _route_clearance, validate_scene
    dims = base["room_dimensions"]
    cases = [_case(route, []), _case(list(reversed(route)), [])]
    h = _digest(dict(boxes=boxes, room_dimensions=dims))
    sc = {k: v for k, v in base.items() if k not in ("start_goals", "routes", "feasibility", "case_feasibility")}
    sc.update(boxes=boxes, geometry_hash=h, scene_id=f"narrow-junk-train-{base['seed']:06d}-{h[:12]}",
              family="narrow", difficulty="junk", start_goals=cases, goal_index=0, **copy.deepcopy(cases[0]))
    sc["routes"] = [dict(case_index=0, reverse_case_index=1, name="narrow", route_length_m=round(cases[0]["route_length_m"], 3))]
    sc["feasibility"] = _route_clearance(route, boxes)   # root cylinder: not met at the pinches, by design
    sc["generator"] = dict(base.get("generator", {}), name=GENERATOR, narrow_profile=profile,
                           description=__doc__.split("\n\n")[0])
    kinds = {}
    for b in boxes:
        if b.get("category") != "wall":
            kinds[b.get("object_type", "?")] = kinds.get(b.get("object_type", "?"), set()) | {b.get("object_id")}
    sc["counts"] = dict(base["counts"], primitive_boxes=len(boxes), generic_objects=sum(len(v) for v in kinds.values()))
    validate_scene(sc)
    return sc


def with_model(scene, fn):
    from cat_ppo.furniture.scenes import _digest
    scene = dict(scene, geometry_hash=_digest(dict(boxes=scene["boxes"], room_dimensions=scene["room_dimensions"])))
    tmp = tempfile.mkdtemp(prefix="narrow_")
    try:
        with open(os.path.join(tmp, "scene.json"), "w") as f:
            json.dump(scene, f)
        return fn(Strategies(tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_room(seed, log=print):
    from .chaotic_rooms import generate_chaotic_room
    register_junk()
    base = generate_chaotic_room(seed, "train", "junk")
    if base["counts"]["tables"] < 1:
        log(f"seed {seed}: no table, skipped")
        return None
    rng = random.Random(seed * 7919 + 17)
    route = base["start_goals"][base["routes"][0]["case_index"]]["route"]
    _, cum = polyline(route)
    L = float(cum[-1])
    boxes = list(base["boxes"])
    samples, s_of = samples_with_s(route)
    # pinch positions about every metre; at least MIN_SIDE sideways ones
    spots = list(np.arange(END_KEEP + rng.uniform(0, .2), L - END_KEEP + 1e-6, .8 + rng.uniform(0, .15)))
    if len(spots) < MIN_PINCHES - 1:
        log(f"seed {seed}: route {L:.1f} m too short for pinches")
        return None
    pinches, protected = [], set()   # objects of kept pinches are never removed

    def want():   # alternate sideways / arms in, starting sideways (the other kind if one fails)
        first = "side" if not pinches or pinches[-1]["kind"] == "arms" else "arms"
        return [first, "arms" if first == "side" else "side"]

    for s in spots:
        for kind in want():
            win = [q[:2] for q, sq in zip(samples, s_of) if abs(sq - s) <= PINCH_HALF]
            keep = [q[:2] for q, sq in zip(samples, s_of) if abs(sq - s) > PINCH_HALF + .45]
            ok = False
            for attempt in range(12):
                gap = rng.uniform(*GAPS[kind])
                new = []
                for side_sign in (1, -1):
                    for _ in range(6):
                        builder = rng.choice(pinch_builders(kind))
                        name = f"pinch{len(pinches)}{'L' if side_sign > 0 else 'R'}{attempt}"
                        parts = builder(rng, name, "train", 900 + len(pinches) * 10 + attempt)
                        placed = place_object(rng, parts, route, s, side_sign, gap / 2, win, keep, base["room_dimensions"],
                                            tall=kind == "side")
                        if placed is not None:
                            new.append(placed)
                            break
                if len(new) != 2:
                    continue
                # clear what the pinch objects sit in (whole objects, never walls)
                lo = np.minimum(*[aabb(p)[0] for p in new]) - .05
                hi = np.maximum(*[aabb(p)[1] for p in new]) + .05
                hit = set()
                for b in boxes:
                    if b.get("category") == "wall":
                        continue
                    blo, bhi = aabb([b])
                    if np.all(blo < hi) and np.all(bhi > lo):
                        hit.add(b.get("object_id") or b["name"])
                if hit & protected:
                    continue
                trial = [b for b in boxes if b.get("category") == "wall" or (b.get("object_id") or b["name"]) not in hit]
                trial += [b for p in new for b in p]
                idx = [i for i, sq in enumerate(s_of) if abs(sq - s) <= PINCH_HALF]
                labs = with_model(dict(base, boxes=trial), lambda chk: [chk.label(*samples[i]) for i in idx])
                if "blocked" in labs:
                    continue
                core = labs[len(labs) // 2 - 2: len(labs) // 2 + 3]
                if kind == "side" and "side" not in core:
                    continue
                if kind == "arms" and ("arms" not in core or "side" in labs):
                    continue
                boxes, ok = trial, True
                protected |= {b.get("object_id") or b["name"] for p_ in new for b in p_}
                pinches.append(dict(kind=kind, route_s_m=round(float(s), 2), gap_m=round(gap, 3), removed_objects=sorted(hit)))
                log(f"seed {seed}: pinch {kind:4s} at {s:.1f} m, gap {gap:.2f} m (attempt {attempt + 1})")
                break
            if ok:
                break
            log(f"seed {seed}: no {kind} pinch at {s:.1f} m")
    # whole route
    labs = with_model(dict(base, boxes=boxes), lambda chk: labels_along(chk, samples))
    rr = runs(labs, s_of)
    n_side = sum(r[0] == "side" for r in rr)
    profile = dict(pinches=pinches, labels={k: round(labs.count(k) / len(labs), 3) for k in ("free", "arms", "side", "blocked")},
                   stretches=[dict(need=r[0], from_m=round(float(r[1]), 2), to_m=round(float(r[2]), 2)) for r in rr],
                   margin_m=MARGIN, shift_m=SHIFT, turn_deg=math.degrees(TURN), gaps_m=GAPS)
    kept_side = sum(p_["kind"] == "side" for p_ in pinches)
    kept_arms = len(pinches) - kept_side
    if "blocked" in labs or n_side < 1 or kept_side < MIN_SIDE or kept_arms < 1 or len(pinches) < MIN_PINCHES:
        log(f"seed {seed}: rejected ({'blocked' if 'blocked' in labs else f'{len(pinches)} pinches ({kept_side} sideways)'})")
        return None
    w = MR.winding(route)
    if math.radians(w["total_turn_deg"]) < MR.MIN_TURN or w["max_detour_m"] < MR.MIN_DETOUR:
        log(f"seed {seed}: rejected (route too straight: {w})")
        return None
    sc = build_scene(base, boxes, route, profile)
    fit, nseg, need = MR.floor_fit(sc, 0)
    if not fit:
        log(f"seed {seed}: rejected (floor {need})")
        return None
    profile.update(w, segments=nseg, floor_need_m=need)
    sc["g1_certificate"] = profile
    log(f"seed {seed}: KEPT {sc['scene_id']}: {len(pinches)} pinches, labels {profile['labels']}, {nseg} segments")
    return sc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="messy_v2")
    p.add_argument("--rooms", type=int, default=4)
    p.add_argument("--max-seeds", type=int, default=40)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "mtc_capture", "scene_sets"))
    a = p.parse_args()
    from .furniture import FurnitureScene
    from .make_scene_set import overview
    root = os.path.abspath(os.path.join(a.out, a.name))
    entries = []
    for seed in range(a.max_seeds):
        if len(entries) >= a.rooms:
            break
        try:
            sc = make_room(seed)
        except RuntimeError as e:
            print(f"seed {seed}: {e}")
            continue
        if sc is None:
            continue
        fs = FurnitureScene(sc, "narrow:junk")
        rel = os.path.join("scenes", "train", "narrow", fs.scene_id, "scene.json")
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        with open(os.path.join(root, rel), "w") as f:
            json.dump(sc, f)
        segs = fs.all_segments(3.0, "auto", (1.32 / MR.TALLEST, *MR.FLOOR))
        entries.append(dict(path=rel, scene_id=fs.scene_id, split="train", family="narrow", variant=f"room {len(entries) + 1}",
                            role=None, seed=seed, route_length_m=round(fs.route_length, 2), hazards=0, routes=1,
                            segments=len(segs), takes_per_segment=3, planned_takes=3 * len(segs),
                            root_route_validated=False, hand_contrast_targets=False,
                            g1_certificate={k: v for k, v in sc["g1_certificate"].items() if k != "pinches"}))
    manifest = dict(schema="mtc-capture-scene-set-v1", name=a.name, segment_length_m=3.0,
                    takes_per_segment=dict(train=3, validation=1, test=0),
                    plan=dict(narrow=dict(rooms=a.rooms, method=__doc__.split("\n\n")[1])),
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
