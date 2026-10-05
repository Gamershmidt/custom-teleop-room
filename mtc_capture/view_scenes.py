"""Look at a scene set in 3D with the branch's own Viser viewer (view_furniture.py): the real
furniture boxes, the native G1 meshes at the route start and the Dex3 hand envelopes.

    # 1. prepare viewer bundles (branch environment: MuJoCo model + native meshes)
    source .venv/bin/activate && source .env
    python -m mtc_capture.view_scenes prepare                     # one scene per family/variant
    python -m mtc_capture.view_scenes prepare --family dense --all  # every dense scene

    # 2. serve them (viewer environment), one port each, and open the printed URLs
    .visual-venv/bin/python -m mtc_capture.view_scenes serve

Bundles go to data/mtc_capture/scene_sets/<name>/viewer/<scene_id>/.
"""

import argparse
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SETS = os.path.join(REPO, "data", "mtc_capture", "scene_sets")


def picks(manifest, family=None, split="train", every=False):
    out, seen = [], set()
    for e in manifest["scenes"]:
        if e["split"] != split or (family and e["family"] != family):
            continue
        key = (e["family"], e["variant"], e["role"])
        if every or key not in seen:
            seen.add(key)
            out.append(e)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["prepare", "serve"])
    p.add_argument("--set", default="messy_v3", help="scene set name under data/mtc_capture/scene_sets")
    p.add_argument("--family", default=None)
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--all", action="store_true", help="every scene, not one per family/variant")
    p.add_argument("--port", type=int, default=8085, help="first port")
    a = p.parse_args()
    sys.path.insert(0, REPO)
    root = os.path.join(SETS, a.set)
    viewer_dir = os.path.join(root, "viewer")

    if a.command == "prepare":
        from view_furniture import prepare
        with open(os.path.join(root, "manifest.json")) as f:
            manifest = json.load(f)
        index = []
        for e in picks(manifest, a.family, a.split, a.all):
            out = os.path.join(viewer_dir, e["scene_id"])
            if not os.path.exists(os.path.join(out, "manifest.json")):   # bundles are reused
                prepare(os.path.dirname(os.path.join(root, e["path"])), out)
            label = " ".join(dict.fromkeys(str(x) for x in (e["family"], e["variant"], e["role"]) if x))
            index.append(dict(bundle=out, label=label, segments=e["segments"], route_m=e["route_length_m"]))
            print(f"prepared {label:40s} {e['scene_id']}")
        with open(os.path.join(viewer_dir, "index.json"), "w") as f:
            json.dump(index, f, indent=1)
        return

    from view_furniture import serve
    with open(os.path.join(viewer_dir, "index.json")) as f:
        index = json.load(f)
    servers = []
    for k, e in enumerate(index):
        port = a.port + k
        servers.append(serve(e["bundle"], port=port))
        print(f"http://127.0.0.1:{port}   {e['label']:40s} {e['segments']} segments, route {e['route_m']} m")
    # every route of multi-route rooms, each in its own colour, with a label at its start
    palette = [(13, 157, 151), (231, 76, 60), (142, 68, 173), (241, 196, 15), (52, 152, 219), (46, 204, 113)]
    for server, e in zip(servers, index):
        with open(os.path.join(e["bundle"], "scene.json")) as f:
            scene = json.load(f)
        if not scene.get("routes"):
            continue
        import numpy as np
        lines = []
        for k, r in enumerate(scene["routes"]):
            route = np.asarray(scene["start_goals"][r["case_index"]]["route"], float)
            pts = np.column_stack([route, np.full(len(route), .03 + .004 * k)])
            server.scene.add_line_segments(f"/routes/{k}", np.stack([pts[:-1], pts[1:]], axis=1),
                                           colors=palette[k % len(palette)], thickness=.03)
            server.scene.add_label(f"/routes/{k}_label", f"route {k + 1}: {r['name']}", position=tuple(pts[0] + [0, 0, .35]))
            if "hand_hazards" in r:
                lines.append(f"- route {k + 1} ({r['name']}): {r['route_length_m']:.1f} m, hand clutter along "
                             f"{r['hand_hazards']['near_fraction']:.0%}")
            else:
                lines.append(f"- route {k + 1} ({r['name']}): {r['route_length_m']:.1f} m")
        cert = scene.get("g1_certificate") or {}
        if cert.get("stretches"):   # narrow rooms: what each stretch of the route needs (narrow_rooms.py)
            route = np.asarray(scene["route"], float)
            cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
            along = lambda s_: np.array([np.interp(s_, cum, route[:, 0]), np.interp(s_, cum, route[:, 1])])
            need_col = {"arms": (243, 156, 18), "side": (231, 76, 60)}
            for j, st in enumerate(cert["stretches"]):
                ss = np.linspace(st["from_m"], st["to_m"], 12)
                q = np.column_stack([along(ss).T, np.full(len(ss), .06)])
                server.scene.add_line_segments(f"/needs/{j}", np.stack([q[:-1], q[1:]], axis=1),
                                               colors=need_col.get(st["need"], (0, 0, 0)), thickness=.09)
            for j, pn in enumerate(cert.get("pinches", [])):
                xy = along(pn["route_s_m"])
                server.scene.add_label(f"/needs/pinch{j}", f"{'SIDEWAYS' if pn['kind'] == 'side' else 'ARMS IN'} "
                                       f"({pn['gap_m']:.2f} m)", position=(float(xy[0]), float(xy[1]), 1.6))
            L = cert["labels"]
            lines.append(f"\n**What the G1 needs along the route** (real collision geometry): walk normally "
                         f"{L['free']:.0%} · arms in / hands up {L['arms']:.0%} (orange) · sideways {L['side']:.0%} (red)")
        with server.gui.add_folder("Routes", expand_by_default=True):
            server.gui.add_markdown(f"**{len(scene['routes'])} routes** (colours in the scene; each is captured)\n\n"
                                    + "\n".join(lines))

    # a Rooms panel in every viewer: previous / next and links to all rooms (one server per room)
    url = lambda i: f"http://127.0.0.1:{a.port + i % len(index)}"
    for k, (server, e) in enumerate(zip(servers, index)):
        rows = "\n".join(f"- {'**' if i == k else ''}[{r['label']}]({url(i)}){'** (here)' if i == k else ''}"
                          for i, r in enumerate(index))
        with server.gui.add_folder("Rooms", expand_by_default=True):
            server.gui.add_markdown(f"**{k + 1}/{len(index)} · {e['label']}**\n\n"
                                    f"[◀ previous]({url(k - 1)}) · [next ▶]({url(k + 1)})\n\n{rows}")
    print("\nDrag to orbit, scroll to zoom; the panel has room/overhead/aisle/robot cameras. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for s in servers:
            s.stop()


if __name__ == "__main__":
    main()
