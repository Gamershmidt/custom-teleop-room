"""Chaotic rooms for hand-protection capture, in the branch's cat-furniture-scene-v1 schema.

The branch's random rooms place whole tables and chairs at random poses, never overlapping,
and search a route afterwards, which usually runs through open floor. Here the room is a mess
at hand height (0.45-1.2 m for an upright G1):
  - tables with clutter on top, often overhanging the edges (boxes, book piles, laptops,
    monitors, planks lying across and sticking out)
  - chairs pushed around at random angles, toppled on their backs, or upside down on a
    table with the backrest hanging over the edge
  - crate stacks with each crate offset and twisted, planks bridging two crates, poles
    lying across a chair and sticking out
  - shelves with things protruding, cabinets with the door open and a drawer out,
    partitions at random angles, floor lamps
Everything is yaw-only oriented boxes (the branch's geometry model), so "toppled" means
laid down along an axis, never tilted.

Routes, reset clearances and validation are the branch's own: random_rooms._admit_routes
(A* on the root cylinder, endpoints with the randomized-reset hand margin), _route_clearance
and scenes.validate_scene. A layout is kept only if it is traversable and the route passes
enough hand-height clutter (hand_hazard_metrics); otherwise it is resampled.

    python -m mtc_capture.chaotic_rooms --seed 0 --difficulty hard --out /tmp/chaos   # needs cat_ppo
"""

import argparse
import copy
import json
import math
import os
import random
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from cat_ppo.furniture.random_rooms import (GRAPH_CLEARANCE_M, RESET_CENTER_CLEARANCE_M, _admit_routes,  # noqa: E402
                                            _clearance_grid, _footprint, _overlap, _search_path, _simplify_path,
                                            transform_object)
from cat_ppo.furniture.scenes import (ROOT_CLEARANCE_RADIUS, ROOT_CLEARANCE_Z, SCHEMA, SPLITS, _box,  # noqa: E402
                                      _case, _chair, _digest, _horizontal_distance, _root_obstacles, _route_clearance,
                                      _table, _walls, validate_scene)

GENERATOR = "mtc-chaotic-hand-clutter-v1"
HAND_ZONE = (0.45, 1.20)
HAZARD_DISTANCE = 0.45   # route centre -> hand-zone object; the G1 hands swing ~0.2-0.3 m out


# ----------------------------------------------------------------- objects (local frame, z from the floor)

def _common(name, kind, split, index):
    return dict(object_id=name, object_type=kind, shape_identity=f"{split}/chaotic/{kind}/{index:03d}")


def _item(rng, name, x, y, z0, common, yaw=0.0):
    """Something lying on a surface at height z0."""
    kind = rng.choice(["box", "box", "books", "laptop", "monitor", "plank"])
    if kind == "box":
        sx, sy, sz = rng.uniform(.15, .45), rng.uniform(.12, .35), rng.uniform(.06, .32)
        return [_box(name, [x, y, z0 + sz / 2], [sx / 2, sy / 2, sz / 2], "item", yaw, **common)]
    if kind == "books":
        parts, z = [], z0
        for k in range(rng.randint(2, 6)):
            sz = rng.uniform(.02, .05)
            parts.append(_box(f"{name}_{k}", [x + rng.uniform(-.03, .03), y + rng.uniform(-.03, .03), z + sz / 2],
                              [rng.uniform(.1, .15), rng.uniform(.07, .11), sz / 2], "item",
                              yaw + rng.uniform(-.5, .5), **common))
            z += sz
        return parts
    if kind == "laptop":
        return [_box(name + "_base", [x, y, z0 + .01], [.17, .12, .01], "item", yaw, **common),
                _box(name + "_lid", [x - .12 * math.cos(yaw), y - .12 * math.sin(yaw), z0 + .12],
                     [.008, .17, .11], "item", yaw, **common)]
    if kind == "monitor":
        return [_box(name + "_stand", [x, y, z0 + .06], [.1, .08, .06], "item", yaw, **common),
                _box(name + "_screen", [x, y, z0 + .12 + .18], [.02, rng.uniform(.25, .35), .18], "item", yaw, **common)]
    length = rng.uniform(.8, 1.5)   # plank / pole lying flat
    return [_box(name, [x, y, z0 + .02], [length / 2, rng.uniform(.02, .09), .02], "plank", yaw, **common)]


def messy_table(rng, name, split, index):
    c = _common(name, "messy_table", split, index)
    w, d, h = rng.uniform(1.0, 1.8), rng.uniform(.6, .9), rng.uniform(.7, .8)
    parts = [dict(p, **c) for p in _table(name, 0, 0, w, d, h, c["shape_identity"])]
    for k in range(rng.randint(2, 6)):
        x, y = rng.uniform(-w / 2 + .1, w / 2 - .1), rng.uniform(-d / 2 + .1, d / 2 - .1)
        if rng.random() < .55:   # overhang an edge: the classic finger hazard
            side = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            x = side[0] * (w / 2 + rng.uniform(-.05, .18)) if side[0] else x
            y = side[1] * (d / 2 + rng.uniform(-.05, .18)) if side[1] else y
        parts += _item(rng, f"{name}_item{k}", x, y, h, c, rng.uniform(-math.pi, math.pi))
    if rng.random() < .3:   # a chair upside down on the table, backrest hanging over the edge
        cw = rng.uniform(.4, .48)
        y0 = d / 2 - .22
        parts += [_box(f"{name}_uchair_seat", [0, y0, h + .028], [cw / 2, .2, .028], "chair_seat", 0, **c),
                  _box(f"{name}_uchair_back", [0, y0 + .2 + .025, h + .056 - .21], [cw / 2, .025, .21], "chair_back", 0, **c)]
        for sx in (-1, 1):
            for sy in (-1, 1):
                parts.append(_box(f"{name}_uchair_leg{sx:+d}{sy:+d}", [sx * (cw / 2 - .045), y0 + sy * .155, h + .056 + .211],
                                  [.023, .023, .211], "chair_leg", 0, **c))
    return parts


def tossed_chair(rng, name, split, index):
    c = _common(name, "chair", split, index)
    return [dict(p, **c) for p in _chair(name, 0, 0, 0, rng.uniform(.4, .5), rng.uniform(.85, 1.0),
                                          rng.random() < .35, c["shape_identity"])]


def toppled_chair(rng, name, split, index):
    """Chair lying on its back: backrest flat on the floor, seat standing, legs sticking out."""
    c = _common(name, "toppled_chair", split, index)
    w = rng.uniform(.4, .5)
    parts = [_box(name + "_back", [0, 0, .025], [w / 2, .22, .025], "chair_back", 0, **c),
             _box(name + "_seat", [0, .245, .22], [w / 2, .028, .2], "chair_seat", 0, **c)]
    for sx in (-1, 1):
        for z in (.06, .38):
            parts.append(_box(f"{name}_leg{sx:+d}{z:.2f}", [sx * (w / 2 - .045), .245 + .211, z], [.023, .211, .023],
                              "chair_leg", 0, **c))
    return parts


def crate_stack(rng, name, split, index):
    c = _common(name, "crate_stack", split, index)
    parts, z = [], 0.0
    for k in range(rng.randint(1, 4)):
        sx, sy, sz = rng.uniform(.35, .6), rng.uniform(.3, .5), rng.uniform(.22, .42)
        if z + sz > 1.45:
            break
        parts.append(_box(f"{name}_{k}", [rng.uniform(-.08, .08), rng.uniform(-.08, .08), z + sz / 2],
                          [sx / 2, sy / 2, sz / 2], "crate", rng.uniform(-.35, .35), **c))
        z += sz
    return parts


def plank_bridge(rng, name, split, index):
    """Two crates with a plank across them, overhanging both ends."""
    c = _common(name, "plank_bridge", split, index)
    gap, h = rng.uniform(.5, 1.1), rng.uniform(.45, .85)
    parts = [_box(f"{name}_crate{s:+d}", [s * (gap / 2 + .2), 0, h / 2], [.2, .22, h / 2], "crate", 0, **c) for s in (-1, 1)]
    length = gap + .8 + rng.uniform(.1, .7)
    parts.append(_box(name + "_plank", [rng.uniform(-.15, .15), rng.uniform(-.1, .1), h + .02],
                      [length / 2, rng.uniform(.05, .12), .02], "plank", rng.uniform(-.25, .25), **c))
    return parts


def pole_on_chair(rng, name, split, index):
    """A broom / pole / pipe lying across a chair seat and backrest, sticking out."""
    parts = tossed_chair(rng, name, split, index)
    c = _common(name, "pole_on_chair", split, index)
    length = rng.uniform(1.0, 1.7)
    parts.append(_box(name + "_pole", [rng.uniform(-.2, .2), -.1, rng.uniform(.5, .9)],
                      [.018, length / 2, .018], "plank", rng.uniform(-.6, .6), **c))
    return parts


def shelf_with_stuff(rng, name, split, index):
    c = _common(name, "shelf", split, index)
    w, d, h = rng.uniform(.8, 1.4), rng.uniform(.3, .45), rng.uniform(1.0, 1.6)
    boards = [.05, *sorted(rng.uniform(.3, h - .15) for _ in range(rng.randint(1, 3))), h - .02]
    parts = [_box(f"{name}_side{s:+d}", [s * (w / 2 - .02), 0, h / 2], [.02, d / 2, h / 2], "support", 0, **c) for s in (-1, 1)]
    parts += [_box(f"{name}_board{k}", [0, 0, z], [w / 2, d / 2, .02], "shelf_edge", 0, **c) for k, z in enumerate(boards)]
    for k, z in enumerate(boards[1:]):
        if rng.random() < .75:   # things sticking out of the shelf front
            sx, sy, sz = rng.uniform(.1, .3), rng.uniform(.15, .4), rng.uniform(.05, .25)
            parts.append(_box(f"{name}_out{k}", [rng.uniform(-w / 2 + .15, w / 2 - .15), d / 2 + sy / 2 - .1, z + .02 + sz / 2],
                              [sx / 2, sy / 2, sz / 2], "item", rng.uniform(-.4, .4), **c))
    return parts


def open_cabinet(rng, name, split, index):
    c = _common(name, "open_cabinet", split, index)
    w, d, h = rng.uniform(.6, 1.0), rng.uniform(.4, .55), rng.uniform(.8, 1.3)
    a = rng.uniform(.6, 1.9)   # door opening angle
    dw = w * rng.choice([.5, 1.0])
    parts = [_box(name + "_body", [0, 0, h / 2], [w / 2, d / 2, h / 2], "cabinet", 0, **c),
             _box(name + "_door", [w / 2 - dw / 2 * math.cos(a), d / 2 + dw / 2 * math.sin(a), h / 2 + .02],
                  [dw / 2, .012, h / 2 - .03], "door", -a, **c)]
    if rng.random() < .6:
        dz, out = rng.uniform(.55, min(h - .1, 1.0)), rng.uniform(.2, .4)
        parts.append(_box(name + "_drawer", [0, d / 2 + out / 2, dz], [w / 2 - .04, out / 2, .07], "cabinet", 0, **c))
    return parts


def partition(rng, name, split, index):
    c = _common(name, "partition", split, index)
    w, h = rng.uniform(.9, 1.6), rng.uniform(1.1, 1.7)
    return [_box(name, [0, 0, h / 2], [w / 2, .03, h / 2], "partition", 0, **c),
            _box(name + "_foot", [0, 0, .02], [w / 2 - .05, .2, .02], "support", 0, **c)]


def floor_lamp(rng, name, split, index):
    c = _common(name, "lamp", split, index)
    h = rng.uniform(1.3, 1.7)
    return [_box(name + "_base", [0, 0, .015], [.15, .15, .015], "support", 0, **c),
            _box(name + "_pole", [0, 0, h / 2], [.015, .015, h / 2], "support", 0, **c),
            _box(name + "_shade", [0, 0, h - .12], [.18, .18, .12], "lamp", 0, **c)]


def pallet(rng, name, split, index):
    """A pallet with boxes stacked on it, some hanging over the edge."""
    c = _common(name, "pallet", split, index)
    w, d = rng.uniform(.9, 1.2), rng.uniform(.7, .9)
    parts = [_box(name + "_base", [0, 0, .07], [w / 2, d / 2, .07], "support", 0, **c)]
    z = .14
    for k in range(rng.randint(1, 4)):
        sx, sy, sz = rng.uniform(.3, .6), rng.uniform(.25, .5), rng.uniform(.2, .4)
        if z + sz > 1.3:
            break
        parts.append(_box(f"{name}_box{k}", [rng.uniform(-w / 2 + .1, w / 2 - .1), rng.uniform(-d / 2, d / 2), z + sz / 2],
                          [sx / 2, sy / 2, sz / 2], "crate", rng.uniform(-.4, .4), **c))
        z += sz * rng.choice([1.0, 1.0, 0.0])   # sometimes side by side instead of on top
    return parts


def plank_pile(rng, name, split, index):
    """Planks and poles crossed over a low support, sticking out in all directions."""
    c = _common(name, "plank_pile", split, index)
    h = rng.uniform(.3, .75)
    parts = [_box(name + "_support", [0, 0, h / 2], [.22, .22, h / 2], "crate", rng.uniform(0, 1.5), **c)]
    z = h
    for k in range(rng.randint(2, 5)):
        length, t = rng.uniform(1.0, 1.9), rng.uniform(.02, .04)
        parts.append(_box(f"{name}_plank{k}", [rng.uniform(-.15, .15), rng.uniform(-.15, .15), z + t],
                          [length / 2, rng.uniform(.02, .1), t], "plank", rng.uniform(-math.pi, math.pi), **c))
        z += 2 * t
    return parts


def floor_junk(rng, name, split, index):
    c = _common(name, "floor_junk", split, index)
    sx, sy, sz = rng.uniform(.2, .5), rng.uniform(.15, .4), rng.uniform(.08, .3)
    return [_box(name, [0, 0, sz / 2], [sx / 2, sy / 2, sz / 2], "item", 0, **c)]


def route_side_hazard(rng, name, split, index):
    """A narrow hand-height thing for the route sides: object local +y faces the route."""
    c = _common(name, "route_side", split, index)
    kind = rng.choice(["crate_tower", "stool_box", "coat_stand", "sign", "plant", "cart_handle"])
    if kind == "crate_tower":
        return crate_stack(rng, name, split, index)
    if kind == "stool_box":   # a stool with a box on it, box overhanging towards the route
        h = rng.uniform(.45, .65)
        return [_box(name + "_stool", [0, 0, h / 2], [.16, .16, h / 2], "support", 0, **c),
                _box(name + "_box", [0, rng.uniform(.05, .15), h + .12], [rng.uniform(.12, .22), .18, .12], "item",
                     rng.uniform(-.3, .3), **c)]
    if kind == "coat_stand":
        parts = [_box(name + "_base", [0, 0, .015], [.18, .18, .015], "support", 0, **c),
                 _box(name + "_pole", [0, 0, .85], [.02, .02, .85], "support", 0, **c)]
        for k in range(rng.randint(1, 3)):   # coats hanging down into the hand zone
            a = rng.uniform(-math.pi, math.pi)
            parts.append(_box(f"{name}_coat{k}", [.12 * math.cos(a), .12 * math.sin(a), 1.2], [.06, .2, .35], "item", a, **c))
        return parts
    if kind == "sign":   # a board on a post, board edge at hand height
        return [_box(name + "_post", [0, -.1, .5], [.025, .025, .5], "support", 0, **c),
                _box(name + "_board", [0, 0, rng.uniform(.75, 1.0)], [rng.uniform(.25, .4), .02, .2], "partition", 0, **c)]
    if kind == "plant":
        hp = rng.uniform(.3, .45)
        return [_box(name + "_pot", [0, 0, hp / 2], [.15, .15, hp / 2], "support", 0, **c),
                _box(name + "_leaves", [0, 0, hp + .3], [.28, .28, .3], "plant", rng.uniform(0, 1.5), **c)]
    h = rng.uniform(.85, 1.0)   # trolley with its handle sticking towards the route
    return [_box(name + "_body", [0, -.1, h / 2], [.3, .22, h / 2], "crate", 0, **c),
            _box(name + "_handle", [0, .16, h + .02], [.25, .04, .02], "plank", 0, **c)]


UPRIGHT_BAND = (0.02, 1.40)   # anything here blocks an upright G1 (feet untracked, head at 1.32 m)


def planning_boxes(boxes):
    """For route planning only: boxes in the upright band but outside the branch's root band
    (0.35-1.05 m: floor junk, toppled-chair legs, shelf tops, lamp shades) are stretched into the
    root band so the root-cylinder planner walks around them instead of over or under them."""
    lo, hi = ROOT_CLEARANCE_Z
    out = []
    for b in boxes:
        z0, z1 = b["center"][2] - b["half_size"][2], b["center"][2] + b["half_size"][2]
        if z1 > UPRIGHT_BAND[0] and z0 < UPRIGHT_BAND[1] and not (z1 >= lo and z0 <= hi):
            b = dict(b, center=[b["center"][0], b["center"][1], (lo + hi) / 2],
                     half_size=[b["half_size"][0], b["half_size"][1], (hi - lo) / 2])
        out.append(b)
    return out


def _route_points(route, spacing=.05):
    pts = []
    for a, b in zip(route, route[1:]):
        n = max(1, math.ceil(math.dist(a, b) / spacing))
        pts += [[a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n] for k in range(n)]
    return pts + [list(route[-1])]


def line_route(rng, boxes, dims, cases, spacing, split, placed, both_sides=False, route=None, keep_clear=None, tag=""):
    """Tier 3 for hand protection: hand-height objects along both route sides, near face
    0.30-0.42 m from the route centre (root cylinder 0.23 m stays clear; G1 hands swing through).
    keep_clear: points of every route in the room, all of which must stay >= 0.29 m clear."""
    route = cases[0]["route"] if route is None else route
    pts = np.array(_route_points(route))
    keep_clear = pts[::2] if keep_clear is None else keep_clear
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    ends = [c["start"][:2] for c in cases] + [c["goal"] for c in cases]
    added, s, k = [], rng.uniform(.9, 1.3), 0
    while s < cum[-1] - .9:
        for side in rng.sample([1, -1], 2) if both_sides else [rng.choice([1, -1])]:
            for _ in range(10):   # kinds, offsets and small shifts along the route
                si = s + rng.uniform(-.2, .2)
                i = int(np.clip(np.searchsorted(cum, si), 0, len(pts) - 1))
                j0, j1 = max(0, i - 3), min(len(pts) - 1, i + 3)
                t = pts[j1] - pts[j0]
                t = t / (np.linalg.norm(t) + 1e-9)
                nrm = np.array([-t[1], t[0]])
                parts = route_side_hazard(rng, f"route_side{tag}_{k:02d}", split, 100 + 20 * len(tag) + k)
                hx, hy = _footprint(parts)
                gap = rng.uniform(.30, .42)
                yaw = math.atan2(-side * nrm[1], -side * nrm[0]) + math.pi / 2   # local +y faces the route
                ctr = pts[i] + side * nrm * (gap + hy)
                if not (hx + .1 < ctr[0] < dims[0] - hx - .1 and hy + .1 < ctr[1] < dims[1] - hy - .1):
                    continue
                fp = (float(ctr[0]), float(ctr[1]), hx, hy, yaw)
                if any(_overlap(fp, o, gap=0.0) for o in placed) or \
                        min(math.dist(e, ctr) for e in ends) <= RESET_CENTER_CLEARANCE_M + max(hx, hy):
                    continue
                world = transform_object(parts, float(ctr[0]), float(ctr[1]), yaw)
                body = [b for b in world if b["center"][2] - b["half_size"][2] < UPRIGHT_BAND[1]]
                if min(_horizontal_distance(p, b) for p in keep_clear for b in body) < .29:
                    continue
                placed.append(fp)
                boxes.extend(world)
                added.append(dict(object_id=parts[0].get("object_id"), kind="route_side", center_xy_m=ctr.tolist(),
                                  yaw_rad=yaw, route_distance_m=round(float(si), 3), side=side, gap_m=round(gap, 3)))
                k += 1
                break
        s += spacing * rng.uniform(.75, 1.25)
    return added


def _near_cell(point, axes, clearance, need, radius=.6):
    """Grid cell closest to point with at least `need` clearance (start/goal pockets)."""
    point = point[:2]
    ix = np.flatnonzero(np.abs(axes[0] - point[0]) <= radius)
    iy = np.flatnonzero(np.abs(axes[1] - point[1]) <= radius)
    best = None
    for i in ix:
        for j in iy:
            if clearance[i, j] >= need:
                d = math.dist((axes[0][i], axes[1][j]), point)
                if d <= radius and (best is None or d < best[0]):
                    best = (d, (int(i), int(j)))
    return None if best is None else best[1]


def route_overlap(a, b, tol=.4):
    """Fraction of route a (sampled every 5 cm) lying within tol of route b."""
    pa, pb = np.array(_route_points(a)), np.array(_route_points(b))
    d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)).min(1)
    return float((d < tol).mean())


def plan_routes(rng, boxes, dims, pockets, max_routes=4, min_length=3.0, max_overlap=.55):
    """Several distinct routes through the same clutter: between different pairs of the reserved
    pockets (opposite, then diagonal), plus alternatives between the same pockets found by blocking
    the middle of the previous path. Same root-cylinder A* and path simplification as the branch;
    planned on the upright-band proxy, so no route goes under or over anything."""
    plan = planning_boxes(boxes)
    axes, clearance = _clearance_grid(plan, dims)
    free = clearance >= GRAPH_CLEARANCE_M
    cells = [_near_cell(q, axes, clearance, RESET_CENTER_CLEARANCE_M) for q in pockets]
    names = [q[2] for q in pockets]
    order = {"W": 0, "E": 1, "S": 2, "N": 3}
    pairs = [("W", "E"), ("S", "N"), ("W", "N"), ("S", "E"), ("W", "S"), ("N", "E")]
    routes = []

    def search(ci, cj, grid):
        path = _search_path(ci, cj, grid, clearance)
        if path is None:
            return None
        r = _simplify_path([[round(float(axes[0][i]), 8), round(float(axes[1][j]), 8)] for i, j in path], plan)
        length = sum(math.dist(p, q) for p, q in zip(r, r[1:]))
        if length < min_length or not _route_clearance(r, boxes)["root_route_validated"] \
                or upright_clearance(boxes, r) < ROOT_CLEARANCE_RADIUS:
            return None
        return r

    def distinct(r):
        return all(route_overlap(r, o) <= max_overlap and route_overlap(o, r) <= max_overlap for o, _ in routes)

    for a, b in pairs:
        if len(routes) >= max_routes:
            break
        if a not in names or b not in names:
            continue
        ci, cj = cells[names.index(a)], cells[names.index(b)]
        if ci is None or cj is None:
            continue
        r = search(ci, cj, free)
        if r is None:
            continue
        if distinct(r):
            routes.append((r, f"{a}-{b}"))
        # an alternative way between the same pockets: block the middle of this path and search again
        if len(routes) < max_routes:
            pts = np.array(_route_points(r, .08))
            mid = pts[int(.25 * len(pts)):int(.75 * len(pts))]
            X, Y = np.meshgrid(axes[0], axes[1], indexing="ij")
            blocked = free.copy()
            for p in mid[::2]:
                blocked &= (X - p[0]) ** 2 + (Y - p[1]) ** 2 > .5 ** 2
            r2 = search(ci, cj, blocked)
            if r2 is not None and distinct(r2):
                routes.append((r2, f"{a}-{b} alt"))
    return routes


OBJECTS = {  # builder, weight (medium), weight (hard)
    "messy_table": (messy_table, 3, 4), "tossed_chair": (tossed_chair, 3, 3), "toppled_chair": (toppled_chair, 1, 2),
    "crate_stack": (crate_stack, 2, 3), "plank_bridge": (plank_bridge, 1, 2), "pole_on_chair": (pole_on_chair, 1, 2),
    "shelf": (shelf_with_stuff, 1, 2), "open_cabinet": (open_cabinet, 1, 2), "partition": (partition, 1, 1),
    "floor_lamp": (floor_lamp, 1, 1), "floor_junk": (floor_junk, 1, 1),
    "pallet": (pallet, 0, 0), "plank_pile": (plank_pile, 0, 0),
}

# Chaotic patterns: object mix, room size, layout and how densely the route is lined.
#   layout: "random" (uniform), "walls" (mostly along the walls, corridor), "rows" (tables in skewed rows)
#   axis:   route axis (None = either); target: hazard coverage a layout must reach (else best of attempts)
_W = lambda col: {k: v[col] for k, v in OBJECTS.items() if v[col]}
PATTERNS = {
    "medium": dict(weights=_W(1), count=(11, 16), dims=((5.0, 6.5), (4.5, 6.0)), layout="random", axis=None,
                   side_spacing=1.1, both_sides=False, target=dict(near_fraction=.45, hazards_per_m=.5)),
    "hard": dict(weights=_W(2), count=(17, 23), dims=((5.5, 7.0), (5.0, 6.5)), layout="random", axis=None,
                 side_spacing=.8, both_sides=True, target=dict(near_fraction=.6, hazards_per_m=.7)),
    "storage": dict(weights=dict(crate_stack=5, pallet=4, shelf=4, plank_bridge=3, plank_pile=2, partition=2,
                                 open_cabinet=1, floor_junk=2),
                    count=(16, 22), dims=((5.0, 6.5), (4.5, 6.0)), layout="random", axis=None,
                    side_spacing=.8, both_sides=True, target=dict(near_fraction=.55, hazards_per_m=.7)),
    "office": dict(weights=dict(messy_table=6, tossed_chair=5, open_cabinet=2, partition=2, floor_lamp=1,
                                toppled_chair=1, shelf=1, pole_on_chair=1),
                   count=(14, 20), dims=((5.0, 6.5), (4.5, 6.0)), layout="random", axis=None,
                   side_spacing=.9, both_sides=True, target=dict(near_fraction=.55, hazards_per_m=.7)),
    "classroom": dict(weights=dict(tossed_chair=5, toppled_chair=3, floor_junk=1, crate_stack=1, pole_on_chair=1),
                      count=(7, 12), dims=((5.5, 7.0), (4.8, 6.0)), layout="rows", axis=None,
                      side_spacing=1.0, both_sides=False, target=dict(near_fraction=.55, hazards_per_m=.6)),
    "corridor": dict(weights=dict(crate_stack=3, shelf=3, open_cabinet=2, messy_table=2, tossed_chair=2,
                                  pole_on_chair=1, plank_bridge=1, pallet=1, floor_lamp=1),
                     count=(10, 16), dims=((8.5, 10.0), (2.6, 3.2)), layout="walls", axis=0, min_routes=1,
                     side_spacing=.9, both_sides=True, target=dict(near_fraction=.6, hazards_per_m=.6)),
    "debris": dict(weights=dict(plank_pile=4, plank_bridge=3, pole_on_chair=3, partition=3, crate_stack=3,
                                toppled_chair=3, pallet=1, floor_junk=2),
                   count=(18, 26), dims=((6.0, 7.5), (5.5, 7.0)), layout="random", axis=None,
                   side_spacing=.8, both_sides=True, target=dict(near_fraction=.6, hazards_per_m=.8)),
}
DIFFICULTIES = tuple(PATTERNS)


# ----------------------------------------------------------------- metrics

def upright_clearance(boxes, route):
    """Minimum horizontal distance from the route centre to anything in the upright band."""
    body = [b for b in boxes if b["center"][2] + b["half_size"][2] > UPRIGHT_BAND[0]
            and b["center"][2] - b["half_size"][2] < UPRIGHT_BAND[1]]
    return min(_horizontal_distance(p, b) for p in _route_points(route, .05) for b in body)


def hand_hazard_metrics(boxes, route, spacing=.05):
    """How much hand-height clutter the route passes: distance from the route centre line to
    hand-zone boxes (not walls), and the objects within HAZARD_DISTANCE."""
    pts = []
    for a, b in zip(route, route[1:]):
        n = max(1, math.ceil(math.dist(a, b) / spacing))
        pts += [[a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n] for k in range(n)]
    pts.append(route[-1])
    hz = [b for b in boxes if b.get("category") != "wall"
          and b["center"][2] - b["half_size"][2] < HAND_ZONE[1] and b["center"][2] + b["half_size"][2] > HAND_ZONE[0]]
    if not hz:
        return dict(hazard_objects=0, hazards_per_m=0.0, near_fraction=0.0, min_hand_zone_distance_m=None)
    d = np.array([[_horizontal_distance(p, b) for b in hz] for p in pts])
    near = d.min(1) < HAZARD_DISTANCE
    objs = {hz[j].get("object_id", hz[j]["name"]) for j in np.flatnonzero(d.min(0) < HAZARD_DISTANCE)}
    length = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
    return dict(hazard_objects=len(objs), hazards_per_m=round(len(objs) / max(length, 1e-6), 3),
                near_fraction=round(float(near.mean()), 3), min_hand_zone_distance_m=round(float(d.min()), 3),
                median_hand_zone_distance_m=round(float(np.median(d.min(1))), 3),
                hazard_distance_m=HAZARD_DISTANCE, hand_zone_m=list(HAND_ZONE))


# ----------------------------------------------------------------- generator

TARGET = {k: v["target"] for k, v in PATTERNS.items()}


def _score(m):
    return m["near_fraction"] + .5 * min(m["hazards_per_m"], 1.5)


def generate_chaotic_room(seed, split="train", difficulty="hard", attempts=250):
    """difficulty: a chaotic pattern, one of PATTERNS (medium, hard, storage, office, classroom, corridor, debris)."""
    if split not in SPLITS or difficulty not in DIFFICULTIES or type(seed) is not int or seed < 0:
        raise ValueError(f"invalid seed/split/pattern; patterns: {DIFFICULTIES}")
    rng = random.Random(int(_digest([GENERATOR, seed, split, difficulty]), 16))
    pat = PATTERNS[difficulty]
    names = list(pat["weights"])
    weights = [pat["weights"][n] for n in names]
    best = None
    for attempt in range(attempts):
        dims = [round(rng.uniform(*pat["dims"][0]), 3), round(rng.uniform(*pat["dims"][1]), 3), 2.4]
        count = rng.randint(*pat["count"])
        objects = []   # (kind, parts, preferred pose or None)
        if pat["layout"] == "rows":   # tables in loose, skewed rows (tables first, the rest scattered)
            nx, ny = max(2, int(dims[0] / 2.1)), max(2, int(dims[1] / 1.9))
            for i in range(nx):
                for j in range(ny):
                    if rng.random() < .8:
                        pose = ((i + .5) * dims[0] / nx + rng.uniform(-.25, .25),
                                (j + .5) * dims[1] / ny + rng.uniform(-.25, .25), rng.uniform(-.45, .45))
                        objects.append(("messy_table", messy_table(rng, f"messy_table_r{i}{j}", split, 50 + i * 10 + j),
                                        pose))
        for i in range(count):
            kind = rng.choices(names, weights)[0]
            objects.append((kind, OBJECTS[kind][0](rng, f"{kind}_{i:02d}", split, i), None))
        objects.sort(key=lambda o: (o[2] is None, -math.prod(_footprint(o[1]))))
        # reserve start/goal pockets in the bands _admit_routes searches (10-25 % / 75-90 % of one axis),
        # sized for the branch's randomized-reset hand margin; the route between them is not reserved
        # four pockets, one near the middle of each wall (corridor: its two ends), for several routes
        axes_used = [pat["axis"]] if pat["axis"] is not None else [0, 1]
        pockets = []
        for axis in axes_used:
            for (lo, hi), tag in zip(((.12, .22), (.78, .88)), ("WE" if axis == 0 else "SN")):
                q = [0.0, 0.0, tag]
                q[axis] = rng.uniform(lo, hi) * dims[axis]
                q[1 - axis] = rng.uniform(.3, .7) * dims[1 - axis]
                pockets.append(q)
        pocket_r = RESET_CENTER_CLEARANCE_M + .1
        placed, kept = [], []
        for kind, parts, pose in objects:
            hx, hy = _footprint(parts)
            for trial in range(400):
                if pose is not None and trial < 60:
                    x, y, yaw = pose[0] + rng.uniform(-.2, .2), pose[1] + rng.uniform(-.2, .2), pose[2] + rng.uniform(-.2, .2)
                else:
                    yaw = rng.uniform(-math.pi, math.pi)
                    if pat["layout"] == "walls" and rng.random() < .8:   # push it against a long wall
                        yaw = rng.choice([0.0, math.pi]) + rng.uniform(-.35, .35)
                c, s_ = abs(math.cos(yaw)), abs(math.sin(yaw))
                ex, ey = c * hx + s_ * hy, s_ * hx + c * hy
                if 2 * ex > dims[0] - .25 or 2 * ey > dims[1] - .25:
                    continue
                if pose is None or trial >= 60:
                    x = rng.uniform(.105 + ex, dims[0] - .105 - ex)
                    y = rng.uniform(.105 + ey, dims[1] - .105 - ey)
                    if pat["layout"] == "walls" and rng.random() < .8:
                        y = rng.choice([.105 + ey + rng.uniform(0, .15), dims[1] - .105 - ey - rng.uniform(0, .15)])
                if not (.105 + ex <= x <= dims[0] - .105 - ex and .105 + ey <= y <= dims[1] - .105 - ey):
                    continue
                fp = (x, y, hx, hy, yaw)
                if any(_overlap(fp, o, gap=0.0) for o in placed):   # objects may touch: it's a mess
                    continue
                if any(_overlap(fp, (q[0], q[1], pocket_r, pocket_r, 0.0), gap=0.0) for q in pockets):
                    continue
                placed.append(fp)
                kept.append((kind, transform_object(parts, x, y, yaw),
                             dict(object_id=parts[0].get("object_id"), kind=kind, center_xy_m=[x, y], yaw_rad=yaw),
                             math.prod(_footprint(parts))))
                break
        count = len(objects)
        # annealed resampling (paper, Sec. III-B.2), aimed at several routes: drop ~15% of the objects,
        # smallest first, until the room has min_routes distinct routes (keeping >= 50% of the layout);
        # the best route set seen is kept if the target is never reached. Tier 3 re-lines the routes after.
        want = pat.get("min_routes", 3)
        admitted, removed, got = None, 0, None
        while len(kept) >= 0.5 * count:
            boxes = _walls(dims) + [b for _, parts, _, _ in kept for b in parts]
            adm = _admit_routes(rng, planning_boxes(boxes), dims)
            if adm is not None:
                routes = plan_routes(rng, boxes, dims, pockets)
                if routes and (got is None or len(routes) > len(got[1])):
                    got = (list(kept), routes, boxes, adm, removed)
                if len(routes) >= want:
                    break
            k = max(1, round(.15 * len(kept)))
            order = sorted(range(len(kept)), key=lambda i: kept[i][3] * rng.uniform(.5, 1.5))
            drop = set(order[:k])
            kept = [o for i, o in enumerate(kept) if i not in drop]
            removed += k
        if got is None:
            continue
        kept, routes, boxes, admitted, removed = got
        boxes = list(boxes)
        profiles = [prof for _, _, prof, _ in kept]
        _, layout = admitted
        kept_xy = {(p["center_xy_m"][0], p["center_xy_m"][1]) for p in profiles}
        placed = [fp for fp in placed if (fp[0], fp[1]) in kept_xy]   # footprints of objects annealing kept
        # tier 3 along every route; each placement keeps all routes clear
        ends_cases = [_case(r, []) for r, _ in routes]
        keep_clear = np.vstack([np.array(_route_points(r))[::2] for r, _ in routes])
        for k, (r, _) in enumerate(routes):
            profiles += line_route(rng, boxes, dims, ends_cases, pat["side_spacing"], split, placed,
                                   both_sides=pat["both_sides"], route=r, keep_clear=keep_clear, tag=f"r{k}")
        # extra, denser passes for routes that still pass too little clutter
        tgt_near = TARGET[difficulty]["near_fraction"]
        for extra in range(3):
            weak = [k for k, (r, _) in enumerate(routes) if hand_hazard_metrics(boxes, r)["near_fraction"] < tgt_near]
            if not weak:
                break
            for k in weak:
                profiles += line_route(rng, boxes, dims, ends_cases, pat["side_spacing"] * (.7 - .1 * extra), split,
                                       placed, both_sides=True, route=routes[k][0], keep_clear=keep_clear,
                                       tag=f"r{k}x{extra}")
        routes = [(r, name) for r, name in routes if _route_clearance(r, boxes)["root_route_validated"]
                  and upright_clearance(boxes, r) >= ROOT_CLEARANCE_RADIUS]
        if not routes:
            continue
        per_route = [hand_hazard_metrics(boxes, r) for r, _ in routes]
        # a route that still passes almost nothing is not worth capturing (keep the best one regardless)
        keep = [k for k, m in enumerate(per_route) if m["near_fraction"] >= .25] or [int(np.argmax([_score(m) for m in per_route]))]
        routes, per_route = [routes[k] for k in keep], [per_route[k] for k in keep]
        order = sorted(range(len(routes)), key=lambda i: -_score(per_route[i]))
        routes, per_route = [routes[i] for i in order], [per_route[i] for i in order]
        if sum(math.dist(p, q) for p, q in zip(routes[0][0], routes[0][0][1:])) < 4.0:
            continue
        metrics = dict(per_route[0])
        metrics["annealing_removed_objects"] = removed
        mean_near = float(np.mean([m["near_fraction"] for m in per_route]))
        cand = (boxes, dims, routes, per_route, layout, metrics, profiles, attempt)
        score = _score(metrics) + .5 * mean_near + .25 * min(len(routes), 4)
        if best is None or score > best[0]:
            best = (score, cand)
        tgt = TARGET[difficulty]
        if len(routes) >= pat.get("min_routes", 3) and all(metrics[k] >= v for k, v in tgt.items()) \
                and mean_near >= tgt["near_fraction"] - .1:
            break
    if best is None:
        raise RuntimeError(f"no traversable chaotic room for seed {seed}")
    boxes, dims, routes, per_route, layout, metrics, profiles, attempt = best[1]
    metrics["meets_target"] = all(metrics[k] >= v for k, v in TARGET[difficulty].items())
    metrics["target"] = TARGET[difficulty]
    # start_goals: every distinct route forwards, then each reversed (the branch's convention keeps both)
    cases = [_case(r, []) for r, _ in routes] + [_case(list(reversed(r)), []) for r, _ in routes]

    geometry_hash = _digest(dict(boxes=boxes, room_dimensions=dims))
    scene = dict(schema=SCHEMA, scene_id=f"chaotic-{difficulty}-{split}-{seed:06d}-{geometry_hash[:12]}",
                 seed=seed, split=split, family="chaotic", difficulty=difficulty, units="metres",
                 coordinate_system="right-handed-z-up", boxes=boxes, room_dimensions=dims,
                 geometry_hash=geometry_hash, start_goals=cases, goal_index=0, **copy.deepcopy(cases[0]))
    kinds = [p["kind"] for p in profiles]
    scene["counts"] = dict(tables=kinds.count("messy_table"),
                           chairs=sum(k in ("tossed_chair", "toppled_chair", "pole_on_chair") for k in kinds),
                           generic_objects=len(kinds), primitive_boxes=len(boxes), bottlenecks=0)
    scene["generator"] = dict(name=GENERATOR, description=__doc__.split("\n\n")[1].strip(),
                              geometry_source="canonical-oriented-boxes", floor_in_obstacle_field=False,
                              shape_profiles=profiles, object_kinds={k: kinds.count(k) for k in sorted(set(kinds))},
                              route_search="several routes between wall pockets and blocked-path alternatives, with "
                                           "cat_ppo.furniture.random_rooms._search_path / _simplify_path (root-cylinder A*) "
                                           "on the upright-band proxy; see scene['routes']",
                              packing_attempt=attempt, full_body_route_certificate=False,
                              time_budget_method="route_length/0.40_m_per_s + 20_s; proposed benchmark budget")
    scene["layout_metrics"] = layout
    metrics["upright_route_clearance_m"] = round(upright_clearance(boxes, scene["route"]), 3)
    metrics["route_side_objects"] = sum(p["kind"] == "route_side" for p in profiles)
    scene["hand_hazards"] = metrics
    scene["routes"] = [dict(case_index=k, reverse_case_index=k + len(routes), name=name,
                            route_length_m=round(cases[k]["route_length_m"], 3), hand_hazards=m)
                       for k, ((_, name), m) in enumerate(zip(routes, per_route))]
    scene["feasibility"] = _route_clearance(scene["route"], boxes)
    scene["case_feasibility"] = [_route_clearance(c["route"], boxes) for c in cases]
    center = min(_horizontal_distance(scene["start"][:2], b) for b in _root_obstacles(boxes))
    scene["reset_clearance"] = dict(root_radius_m=ROOT_CLEARANCE_RADIUS, root_z_interval_m=list(ROOT_CLEARANCE_Z),
                                    center_clearance_m=center, required_center_clearance_m=RESET_CENTER_CLEARANCE_M,
                                    root_cylinder_validated=True, full_body_validated=False,
                                    runtime_nominal_pose_collision_check_required=True)
    validate_scene(scene)
    return scene


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--split", default="train", choices=SPLITS)
    p.add_argument("--difficulty", default="hard", choices=DIFFICULTIES)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    for seed in range(a.seed, a.seed + a.n):
        s = generate_chaotic_room(seed, a.split, a.difficulty)
        d = os.path.join(a.out, s["scene_id"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "scene.json"), "w") as f:
            json.dump(s, f)
        print(d, s["counts"], "route", round(s["route_length_m"], 1), "m", s["hand_hazards"])


if __name__ == "__main__":
    main()
