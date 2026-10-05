"""Click-and-Traverse furniture scenes (cat-furniture-scene-v1) for VR capture.

Scenes come from this branch's own generators (cat_ppo.furniture: dense tables and chairs,
random rooms, generic clutter, hand passages) or from any scene.json they wrote, so the
operator walks exactly the geometry the policy trains on. Frame and units are the
scene's: metres, z up, origin at the room corner, obstacles are yaw-oriented boxes.

A dense room's route is ~40 m, far more than any tracked floor. Capture therefore works on
**segments**: windows of the route (default 3 m at G1 scale, ~4 m of real floor) centred on
each bottleneck, or consecutive pieces of routes without bottlenecks. Every segment is
re-anchored so its start lies on the operator's fixed home spot.
"""

import copy
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

SCHEMA = "cat-furniture-scene-v1"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

COLORS = {  # close to view_furniture.py
    "wall": "#d9d6cf", "tabletop": "#c8834a", "table_leg": "#3b3b3b", "chair_seat": "#1b7a8c",
    "chair_back": "#1b7a8c", "chair_leg": "#2a2a2a", "chair_armrest": "#145a68", "overhead": "#ffb347",
    "overhead_support": "#9a9a9a", "shelf_edge": "#a0784c", "shelf_support": "#555555",
    "item": "#d9534f", "plank": "#caa472", "crate": "#8d6e4f", "support": "#5c5c5c", "cabinet": "#7f8c8d",
    "door": "#b08968", "partition": "#95a5a6", "lamp": "#f4e3a1", "plant": "#4d8a45",
}
DEFAULT_COLOR = "#9c8f7a"

@dataclass
class Segment:
    index: int
    s0: float                      # route distance of the start [m]
    s1: float
    route: np.ndarray              # (k, 2) polyline in the scene frame
    start: np.ndarray              # (3,) x, y, yaw
    goal: np.ndarray               # (2,)
    bottlenecks: list = field(default_factory=list)
    route_id: str = "0"            # which route of the scene: start_goals case index, "r" suffix = walked backwards
    view: object = None            # the FurnitureScene route view this segment belongs to

    @property
    def length(self):
        return self.s1 - self.s0

    def to_json(self):
        return dict(index=self.index, route_id=self.route_id, route_s0_m=round(self.s0, 4), route_s1_m=round(self.s1, 4),
                    route=np.round(self.route, 4).tolist(), start=np.round(self.start, 5).tolist(),
                    goal=np.round(self.goal, 4).tolist(), bottlenecks=self.bottlenecks)


class FurnitureScene:
    def __init__(self, data, source=None):
        if data.get("schema") != SCHEMA:
            raise ValueError(f"{source}: expected schema {SCHEMA}, got {data.get('schema')}")
        self.data = data
        self.source = source
        self.scene_id = data["scene_id"]
        self.dims = np.asarray(data["room_dimensions"], float)
        self.boxes = data["boxes"]
        self.route = np.asarray(data["route"], float)
        self.bottlenecks = data.get("bottlenecks", [])
        self.goal_radius = 0.3
        # one asset per object (a whole chair lights up), walls individually
        self.asset_of_box, self.labels = [], {}
        ids = {}
        for b in self.boxes:
            key = b.get("furniture_id") or b.get("object_id") or b["name"]   # whole objects light up together
            if key not in ids:
                ids[key] = len(ids)
                self.labels[ids[key]] = key
            self.asset_of_box.append(ids[key])
        seg = np.linalg.norm(np.diff(self.route, axis=0), axis=1)
        self.cum = np.r_[0.0, np.cumsum(seg)]
        self.sha1 = hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()

        self.route_id, self.contrast = str(data.get("goal_index", 0)), "forward"

    @property
    def route_length(self):
        return float(self.cum[-1])

    def _view(self, route, bottlenecks, route_id, contrast):
        v = copy.copy(self)
        v.route = np.asarray(route, float)
        v.cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(v.route, axis=0), axis=1))]
        v.bottlenecks, v.route_id, v.contrast = bottlenecks, route_id, contrast
        return v

    def route_views(self, reverse=False):
        """The routes to capture. Scenes listing several distinct routes (scene['routes'], chaotic
        rooms) give one view per route; other scenes their selected route. reverse=True adds each
        route walked backwards (hand-contrast zones and gates mirrored)."""
        cases = self.data.get("start_goals") or []
        if self.data.get("routes"):
            picks = [(r["case_index"], r.get("reverse_case_index")) for r in self.data["routes"]]
        else:
            picks = [(self.data.get("goal_index", 0), None)]
        views = []
        for k, rk in picks:
            case = cases[k] if k < len(cases) else dict(route=self.data["route"], bottlenecks=self.bottlenecks)
            first = k == self.data.get("goal_index", 0)
            views.append(self._view(case["route"], case.get("bottlenecks", []), str(k), "forward" if first else None))
            if reverse:
                route = list(reversed(case["route"]))
                length = float(np.sum(np.linalg.norm(np.diff(np.asarray(route), axis=0), axis=1)))
                gates = [dict(g, route_distance_m=round(length - g["route_distance_m"], 8))
                         for g in case.get("bottlenecks", []) if g.get("route_distance_m") is not None]
                views.append(self._view(route, gates, f"{k}r", "reverse" if first else None))
        return views

    def pretty(self, asset_id):
        """Readable object name for the operator: 'messy_table_r12' -> 'messy table'."""
        import re
        name = self.labels.get(asset_id, "?")
        name = re.sub(r"(_\d+)+$", "", name)            # trailing indices
        name = re.sub(r"_?r\d+(x\d+)?$", "", name)       # route tags of route-side objects
        return re.sub(r"_+", " ", name).strip() or "?"

    def obstacle_arrays(self):
        c = np.array([b["center"] for b in self.boxes], float)
        h = np.array([b["half_size"] for b in self.boxes], float)
        yaw = np.array([b.get("yaw", 0.0) for b in self.boxes], float)
        return c, h, yaw, np.array(self.asset_of_box, int)

    def point_at(self, s):
        s = float(np.clip(s, 0, self.cum[-1]))
        i = int(min(np.searchsorted(self.cum, s, side="right") - 1, len(self.route) - 2))
        u = (s - self.cum[i]) / max(self.cum[i + 1] - self.cum[i], 1e-12)
        d = self.route[i + 1] - self.route[i]
        return self.route[i] + u * d, math.atan2(d[1], d[0])

    def sub_route(self, s0, s1):
        inner = [self.route[i] for i in range(len(self.route)) if s0 < self.cum[i] < s1]
        return np.array([self.point_at(s0)[0], *inner, self.point_at(s1)[0]])

    def hazards(self):
        """Route distances to capture around: bottleneck gates, or the hand_contrast hazard
        zones of the table-edge / contrastive passages. -> list of (id, distance, info)."""
        out = [(g["id"], g["route_distance_m"], dict(kind="bottleneck", required_behavior=g.get("required_behavior")))
               for g in self.bottlenecks if g.get("route_distance_m") is not None]
        hc = (self.data.get("hand_contrast") or {}) if self.contrast else {}
        for k, z in enumerate(hc.get("zones", [])):
            if "hazard_start_m" in z:
                mid = (z["hazard_start_m"] + z["hazard_end_m"]) / 2
                out.append((f"hazard_{k}", mid if self.contrast == "forward" else self.route_length - mid,
                            dict(kind="hand_contrast_zone", role=hc.get("role"), hazard_m=[z["hazard_start_m"], z["hazard_end_m"]],
                                 hand_active=z.get("hand_active"), forward_weight=z.get("forward_weight"))))
        return sorted(out, key=lambda h: h[1])

    def segments(self, length=3.0, mode="auto", fit=None):
        """Route windows: centred on each hazard ('gates': bottlenecks or hand-contrast zones),
        consecutive pieces ('split'), or the whole route ('full'). 'auto' = gates if the scene
        has hazards, else split. Routes up to 1.25 x length stay whole. fit=(alpha, fwd_max,
        width_max) halves windows until they fit that much real floor (down to 1.2 m of route).
        Each segment starts facing along its start->end line."""
        total = self.route_length
        hazards = self.hazards()
        if mode == "auto":
            mode = "gates" if hazards else "split"
        if mode == "full" or total <= 1.25 * length:
            spans = [(0.0, total)]
        elif mode == "gates":
            spans = []
            for _, d, _ in hazards:
                s0 = min(max(0.0, d - length / 2), max(0.0, total - length))
                spans.append((s0, min(total, s0 + length)))
        else:
            n = math.ceil(total / length)
            spans = [(total * k / n, total * (k + 1) / n) for k in range(n)]
        if fit is not None:   # split windows that need more real floor than there is
            spans = [piece for span in spans for piece in self._fit(span, *fit)]
        out = []
        for k, (s0, s1) in enumerate(spans):
            route = self.sub_route(s0, s1)
            chord = route[-1] - route[0]
            yaw = math.atan2(chord[1], chord[0])   # start facing the segment's end: least sideways floor
            inside = [dict(id=i, route_distance_m=round(d, 4), **info) for i, d, info in hazards if s0 <= d <= s1]
            out.append(Segment(k, s0, s1, route, np.r_[route[0], yaw], route[-1], inside, self.route_id, self))
        return out

    def all_segments(self, length=3.0, mode="auto", fit=None, reverse=False):
        """Segments of every route to capture (see route_views), numbered through."""
        out = []
        for v in self.route_views(reverse):
            for seg in v.segments(length, mode, fit):
                seg.index = len(out)
                out.append(seg)
        return out

    def floor_need(self, s0, s1, alpha):
        """Real floor [m] a segment needs: (length along its start->end line incl. anything behind the
        start, full width) at scale 1/alpha. G1-scale margins: 0.3 m behind, 0.4 m ahead, 0.5 m each side."""
        route = self.sub_route(s0, s1)
        chord = route[-1] - route[0]
        c, s = math.cos(-math.atan2(chord[1], chord[0])), math.sin(-math.atan2(chord[1], chord[0]))
        loc = (route - route[0]) @ np.array([[c, s], [-s, c]])
        fwd = loc[:, 0].max() - min(0.0, loc[:, 0].min()) + 0.7   # 0.3 m behind the start pad, 0.4 m past the goal
        width = 2 * (np.abs(loc[:, 1]).max() + 0.5)
        return fwd / alpha, width / alpha

    def _fit(self, span, alpha, fwd_max, width_max, min_len=1.2):
        s0, s1 = span
        fwd, width = self.floor_need(s0, s1, alpha)
        if (fwd <= fwd_max and width <= width_max) or s1 - s0 < 2 * min_len:
            return [span]
        mid = (s0 + s1) / 2
        return self._fit((s0, mid), alpha, fwd_max, width_max, min_len) + self._fit((mid, s1), alpha, fwd_max, width_max, min_len)

    def save_copy(self, scenes_dir):
        d = os.path.join(scenes_dir, self.scene_id)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "scene.json")
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(self.data, f)
        return path


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "scene.json")
    with open(path) as f:
        return FurnitureScene(json.load(f), os.path.abspath(path))


def find_scenes(root):
    out = []
    for d, _, files in sorted(os.walk(root)):
        if "scene.json" in files:
            try:
                with open(os.path.join(d, "scene.json")) as f:
                    if json.load(f).get("schema") == SCHEMA:
                        out.append(os.path.join(d, "scene.json"))
            except (OSError, ValueError):
                pass
    return out


def generate(spec, seed, split="train"):
    """spec: dense[:family] | pilot[:family] | open | random[:furniture|generic_clutter] |
    clutter[:family] | chaotic[:medium|hard] | hand_table_aisle[:easy|medium|hard] | hand_shelf_passage[:difficulty]"""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    kind, _, variant = spec.partition(":")
    if kind in ("dense", "pilot", "open"):
        from cat_ppo.furniture.scenes import generate_scene
        return FurnitureScene(generate_scene(seed, split, variant or "mixed",
                                             {"open": "open_floor"}.get(kind, kind)), f"generated:{spec}")
    if kind == "chaotic":
        from .chaotic_rooms import generate_chaotic_room
        return FurnitureScene(generate_chaotic_room(seed, split, variant or "hard"), f"generated:{spec}")
    if kind == "random":
        from cat_ppo.furniture.random_rooms import generate_random_room
        return FurnitureScene(generate_random_room(seed, split, variant or "furniture"), f"generated:{spec}")
    if kind == "clutter":
        from cat_ppo.furniture.clutter import generate_clutter_scene
        return FurnitureScene(generate_clutter_scene(seed, split, variant or "mixed"), f"generated:{spec}")
    if kind in ("hand_table_aisle", "hand_shelf_passage"):
        from cat_ppo.furniture.hand_passages import generate_hand_passage_scene
        try:
            d = generate_hand_passage_scene(seed, split, kind, variant or "easy", certify=True)
        except ImportError as e:   # certification needs scikit-fmm; geometry does not
            print(f"[scene] {spec}: certificate skipped ({e}); geometry is the same")
            d = generate_hand_passage_scene(seed, split, kind, variant or "easy", certify=False)
        return FurnitureScene(copy.deepcopy(d), f"generated:{spec}")
    raise SystemExit(f"unknown --generate kind {kind!r}; table-edge/contrastive passages need the branch's "
                     "JAX environment: generate them there and pass --scenes DIR")
