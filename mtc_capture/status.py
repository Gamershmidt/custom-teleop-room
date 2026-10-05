"""Collection progress and quality, from the takes on disk.

    python -m mtc_capture.status                    # all operators
    python -m mtc_capture.status --operator alice   # one operator (--per-operator collections)
    python -m mtc_capture.status --json status.json

Per family: successful / planned takes, how many were safe, the other outcomes, recorded minutes,
hand-tracking coverage, and which objects the G1 proxy touched most. Planned takes use the same
segmentation as capture (operator height and --space); segments of other heights still count
wherever their route interval matches.
"""

import argparse
import json
import os
from collections import Counter, defaultdict

from . import furniture
from .capture import G1_HEIGHT, segment_key

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", default=os.path.join(REPO, "data", "mtc_capture", "scene_sets", "messy_v3", "manifest.json"))
    p.add_argument("--out", default=os.path.join(REPO, "data", "mtc_capture"))
    p.add_argument("--operator", default=None)
    p.add_argument("--operator-height", type=float, default=1.75, help="for planned segmentation")
    p.add_argument("--space", type=float, nargs=2, default=[5.5, 3.0])
    p.add_argument("--reverse", action="store_true", help="plan reversed routes too (as capture --reverse)")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    with open(a.scenes) as f:
        m = json.load(f)
    base = os.path.dirname(os.path.abspath(a.scenes))
    family_of = {e["scene_id"]: (e["family"], e["split"]) for e in m["scenes"]}
    alpha = G1_HEIGHT / a.operator_height
    plan = defaultdict(lambda: dict(segments=0, takes=0))
    for e in m["scenes"]:
        if e["takes_per_segment"]:
            sc = furniture.load(os.path.join(base, e["path"]))
            n = len(sc.all_segments(3.0, "auto", (alpha, *a.space), reverse=a.reverse))
            plan[(e["family"], e["split"])]["segments"] += n
            plan[(e["family"], e["split"])]["takes"] += n * e["takes_per_segment"]

    stats = defaultdict(lambda: dict(success=0, safe=0, other=Counter(), seconds=0.0, tracked=[], touched=Counter(),
                                     operators=Counter()))
    takes_dir = os.path.join(a.out, "takes")
    for d in sorted(os.listdir(takes_dir)) if os.path.isdir(takes_dir) else []:
        try:
            with open(os.path.join(takes_dir, d, "meta.json")) as f:
                t = json.load(f)
        except (OSError, ValueError):
            continue
        if a.operator and t.get("operator") != a.operator or t["scene_id"] not in family_of:
            continue
        s = stats[family_of[t["scene_id"]]]
        s["operators"][t["operator"]] += 1
        s["seconds"] += t.get("duration_s") or 0
        if t["status"] == "success":
            s["success"] += 1
            s["safe"] += bool(t["safe"])
            s["tracked"].append(min(t["tracking_coverage"].values()))
            for obj in t.get("assets_touched", []):
                s["touched"][obj["object"].rsplit("_", 1)[0] if obj["object"][-1].isdigit() else obj["object"]] += 1
        else:
            s["other"][t["status"]] += 1

    rows = []
    print(f"{'split':10s} {'family':18s} {'takes ok/planned':>17s} {'safe':>6s} {'other':>18s} {'min':>6s} {'tracked':>8s}  touched most")
    for (fam, split), pl in sorted(plan.items(), key=lambda kv: (kv[0][1] != "train", kv[0][0])):
        s = stats[(fam, split)]
        safe = f"{s['safe'] / s['success']:.0%}" if s["success"] else "-"
        other = ", ".join(f"{k} {v}" for k, v in s["other"].items()) or "-"
        tr = f"{sum(s['tracked']) / len(s['tracked']):.0%}" if s["tracked"] else "-"
        top = ", ".join(f"{k} {v}" for k, v in s["touched"].most_common(3)) or "-"
        print(f"{split:10s} {fam:18s} {s['success']:>8d}/{pl['takes']:<8d} {safe:>6s} {other:>18s} {s['seconds'] / 60:>6.1f} {tr:>8s}  {top}")
        rows.append(dict(split=split, family=fam, planned_segments=pl["segments"], planned_takes=pl["takes"],
                         success=s["success"], safe=s["safe"], other=dict(s["other"]), minutes=round(s["seconds"] / 60, 2),
                         operators=dict(s["operators"]), touched=dict(s["touched"])))
    tot_ok, tot_plan = sum(r["success"] for r in rows), sum(r["planned_takes"] for r in rows)
    tot_safe = sum(r["safe"] for r in rows)
    ops = Counter()
    for r in rows:
        ops.update(r["operators"])
    print(f"\ntotal: {tot_ok}/{tot_plan} successful takes ({tot_safe} safe), "
          f"{sum(r['minutes'] for r in rows):.0f} min recorded, operators {dict(ops) or '-'}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(rows=rows, total_success=tot_ok, total_planned=tot_plan, total_safe=tot_safe), f, indent=1)


if __name__ == "__main__":
    main()
