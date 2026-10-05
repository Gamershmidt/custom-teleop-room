"""Build the scene set for hand-protection capture from this branch's generators.

Run in the branch environment (it needs JAX and scikit-fmm for the certified passages):

    source .venv/bin/activate && source .env
    python -m mtc_capture.make_scene_set                       # the default "hands" plan
    python -m mtc_capture.make_scene_set --scale 0.3 --name pilot   # a small pilot set first

Output, data/mtc_capture/scene_sets/<name>/:
    scenes/<split>/<family>/<scene_id>/scene.json   cat-furniture-scene-v1, unchanged
    manifest.json      every scene with split, family, seed, segments and planned takes; its
                       "capture_order" interleaves families so a partial capture stays balanced
    overview.png       one top view per family

Then capture (teleop environment):
    python -m mtc_capture.capture --scenes data/mtc_capture/scene_sets/<name>/manifest.json --operator NAME --operator-height H

Splits use the generators' own disjoint shape profiles (train/validation/test). Test scenes
are for evaluating policies in simulation; the plan captures none there.
"""

import argparse
import json
import math
import os
from collections import Counter

# family -> (variants, seeds per variant per split). Seeds are counted per variant, so chaotic
# with 7 patterns x 12 seeds = 84 training scenes.
PLAN = {
    #  family                variants (roles for table_edge / contrastive)                       train  val  test
    "hand_table_aisle":   (["medium"],                                                            10,   2,   2),
    "hand_shelf_passage": (["medium"],                                                            10,   2,   2),
    "table_edge":         (["forward_protected"],                                                 10,   2,   2),
    "contrastive":        (["narrow"],                                                             8,   1,   2),
    "dense":              (["table"],                                                              4,   1,   1),
    "random":             (["furniture", "generic_clutter"],                                       8,   1,   2),
    "chaotic":            (["medium", "hard", "storage", "office", "classroom", "corridor", "debris"], 12,   2,   2),
}
ROLE_FAMILIES = ("table_edge", "contrastive")   # their generators build all roles of a layout together
TAKES = {"train": 3, "validation": 1, "test": 0}   # takes per segment


def generate(family, variant, seed, split):
    """-> list of scene dicts (pair/group generators return several)."""
    if family in ("hand_table_aisle", "hand_shelf_passage"):
        from cat_ppo.furniture.hand_passages import generate_hand_passage_scene
        return [generate_hand_passage_scene(seed, split, family, variant, certify=True)]
    if family == "table_edge":
        from cat_ppo.furniture.table_edge_passages import generate_table_edge_pair
        return [d for d in generate_table_edge_pair(seed, split, certify=True) if d["hand_contrast"]["role"] == variant]
    if family == "contrastive":
        from cat_ppo.furniture.contrastive_passages import generate_contrastive_group
        return [d for d in generate_contrastive_group(seed, split, certify=True) if d["hand_contrast"]["role"] == variant]
    if family in ("dense", "pilot"):
        from cat_ppo.furniture.scenes import generate_scene
        return [generate_scene(seed, split, variant, family)]
    if family == "chaotic":
        from .chaotic_rooms import generate_chaotic_room
        return [generate_chaotic_room(seed, split, variant)]
    if family == "random":
        from cat_ppo.furniture.random_rooms import generate_random_room
        return [generate_random_room(seed, split, variant)]
    raise ValueError(family)


def capture_order(entries):
    """Round-robin over every family/variant (chaotic patterns, roles), train then validation, so
    stopping early stays balanced and consecutive room numbers show different kinds of room."""
    order = []
    for split in ("train", "validation"):
        groups = {}
        for e in entries:
            if e["split"] == split:
                groups.setdefault((e["family"], e["variant"] or e["role"]), []).append(e["path"])
        while any(groups.values()):
            for key in list(groups):
                if groups[key]:
                    order.append(groups[key].pop(0))
    return order


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="hands_v2")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "mtc_capture", "scene_sets"))
    p.add_argument("--scale", type=float, default=1.0, help="multiply seeds per family (0.3 = pilot)")
    p.add_argument("--segment-length", type=float, default=3.0, help="must match capture --segment-length")
    p.add_argument("--first-seed", type=int, default=0)
    a = p.parse_args()

    from .furniture import FurnitureScene

    root = os.path.abspath(os.path.join(a.out, a.name))
    entries = []
    for family, (variants, *per_split) in PLAN.items():
        for split, n in zip(("train", "validation", "test"), per_split):
            n = max(1, round(n * a.scale)) if n else 0
            for variant in variants:
                for seed in range(a.first_seed, a.first_seed + n):
                    for d in generate(family, variant, seed, split):
                        sc = FurnitureScene(d, f"{family}:{variant}")
                        role = (d.get("hand_contrast") or {}).get("role")
                        rel = os.path.join("scenes", split, family, sc.scene_id, "scene.json")
                        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
                        with open(os.path.join(root, rel), "w") as f:
                            json.dump(d, f)
                        segs = sc.all_segments(a.segment_length, "auto")
                        entries.append(dict(
                            path=rel, scene_id=sc.scene_id, split=split, family=family, variant=variant, role=role,
                            seed=seed, route_length_m=round(sc.route_length, 2), hazards=len(sc.hazards()),
                            routes=len(sc.route_views()), segments=len(segs), takes_per_segment=TAKES[split],
                            planned_takes=len(segs) * TAKES[split],
                            root_route_validated=bool((d.get("feasibility") or {}).get("root_route_validated")),
                            hand_contrast_targets="hand_contrast" in d))
                        print(f"{split:10s} {family:18s} {str(variant or role):18s} {sc.scene_id[:48]:48s} "
                              f"{len(segs)} seg")

    manifest_order = capture_order(entries)
    summary = {}
    for split in ("train", "validation", "test"):
        es = [e for e in entries if e["split"] == split]
        summary[split] = dict(scenes=len(es), segments=sum(e["segments"] for e in es),
                              planned_takes=sum(e["planned_takes"] for e in es),
                              by_family=dict(Counter(e["family"] for e in es)))
    manifest = dict(schema="mtc-capture-scene-set-v1", name=a.name, segment_length_m=a.segment_length,
                    takes_per_segment=TAKES, plan={k: dict(variants=v[0], seeds=v[1:]) for k, v in PLAN.items()},
                    summary=summary, capture_order=manifest_order, scenes=entries)
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    overview(root, entries)
    print(json.dumps(summary, indent=1))
    print(os.path.join(root, "manifest.json"))


def overview(root, entries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    from .furniture import COLORS, DEFAULT_COLOR, load

    picks = []
    for e in entries:
        key = (e["family"], e["variant"] if e["family"] != "contrastive" else e["role"])
        if e["split"] == "train" and key not in [k for k, _ in picks]:
            picks.append((key, e))
    n = len(picks)
    cols = 4
    fig, axes = plt.subplots(math.ceil(n / cols), cols, figsize=(4.2 * cols, 3.8 * math.ceil(n / cols)))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, ((fam, var), e) in zip(axes.ravel(), picks):
        sc = load(os.path.join(root, e["path"]))
        for b in sorted(sc.boxes, key=lambda b: b["center"][2] + b["half_size"][2]):
            c, h, y = b["center"], b["half_size"], b.get("yaw", 0.0)
            R = [[math.cos(y), -math.sin(y)], [math.sin(y), math.cos(y)]]
            pts = [(c[0] + R[0][0] * h[0] * sx + R[0][1] * h[1] * sy, c[1] + R[1][0] * h[0] * sx + R[1][1] * h[1] * sy)
                   for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
            ax.add_patch(Polygon(pts, closed=True, fc=COLORS.get(b.get("category", ""), DEFAULT_COLOR), ec="k", lw=0.2,
                                 alpha=0.9 if b.get("category") != "wall" else 0.4))
        ax.plot(sc.route[:, 0], sc.route[:, 1], color="#11a3a3", lw=1.5)
        for g in sc.bottlenecks:
            ax.plot(*g["center"], "o", color="#e8141d", ms=3)
        ax.set_xlim(-0.2, sc.dims[0] + 0.2), ax.set_ylim(-0.2, sc.dims[1] + 0.2), ax.set_aspect("equal")
        ax.set_title(f"{fam} {var or ''}\n{e['segments']} segments, route {e['route_length_m']} m", fontsize=9)
    fig.suptitle("Scene set: one training example per family/variant (teal = route, red = gates)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(root, "overview.png"), dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
