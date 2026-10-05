"""Render a recorded take as a video: top and side view of the furniture around the take's
route segment (segment frame: start at the origin, x along the route), the head path, and the
G1-sized hand envelopes / forearms coloured by clearance (blue ok, orange within the margin,
red contact). For checking takes; the take files are not changed.

    python -m mtc_capture.replay data/mtc_capture/takes/<take_id>            # -> <take_id>/replay.mp4
    python -m mtc_capture.replay <take_dir> --out /tmp/take.gif
"""

import argparse
import json
import math
import os

import numpy as np

from . import clearance as C
from . import furniture

SIDES = ("left", "right")
HAND_ZONE = (0.45, 1.20)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("take")
    p.add_argument("--out", default=None, help=".mp4 (needs ffmpeg) or .gif")
    p.add_argument("--fps", type=float, default=25)
    p.add_argument("--speed", type=float, default=1.0, help="playback speed")
    p.add_argument("--margin", type=float, default=1.3, help="view around the segment route [m]")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Circle, Polygon

    meta = json.load(open(os.path.join(a.take, "meta.json")))
    z = np.load(os.path.join(a.take, "motion.npz"))
    root = os.path.dirname(os.path.dirname(os.path.abspath(a.take)))
    scene = furniture.load(os.path.join(root, meta["scene_file"]))
    obst = C.ObstacleSet(scene)
    seg = meta["segment"]
    Ti, margin = np.array(meta["T_scene_from_xr"]), meta["hand_margin"]
    H = C.home_from_scene(seg["start"])            # scene -> segment frame, for display
    Ti_home = H @ Ti
    controllers = meta["input"] == "controllers"
    t = z["t"]
    head = C.to_scene(Ti_home, z["head_xr"][:, :3, 3])
    route = C.to_scene(H, np.c_[np.array(seg["route"]), np.zeros(len(seg["route"]))])
    lo, hi = route[:, :2].min(0) - a.margin, route[:, :2].max(0) + a.margin

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 9.5), gridspec_kw=dict(height_ratios=[2.2, 1]))
    yaw0 = -seg["start"][2]
    for b in sorted(scene.boxes, key=lambda b: b["center"][2] + b["half_size"][2]):
        c = C.to_scene(H, np.array(b["center"]))
        yaw = b.get("yaw", 0.0) + yaw0
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        pts = [c[:2] + R @ (np.array(b["half_size"][:2]) * s) for s in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        q = np.array(pts)
        if np.any(q.max(0) < lo - 0.5) or np.any(q.min(0) > hi + 0.5):
            continue
        zlo, zhi = c[2] - b["half_size"][2], c[2] + b["half_size"][2]
        cat = b.get("category", "")
        col = furniture.COLORS.get(cat, furniture.DEFAULT_COLOR)
        hand = zlo < HAND_ZONE[1] and zhi > HAND_ZONE[0] and cat != "wall"
        ax.add_patch(Polygon(pts, closed=True, fc=col, ec="k", lw=0.3, alpha=0.85 if hand else 0.35))
        if cat != "wall" and q[:, 1].min() < 1.0 and q[:, 1].max() > -1.0:   # side view: near the start line
            bx.add_patch(Polygon([(q[:, 0].min(), zlo), (q[:, 0].max(), zlo), (q[:, 0].max(), zhi),
                                  (q[:, 0].min(), zhi)], closed=True, fc=col, ec="k", lw=0.2, alpha=0.3))
    ax.plot(route[:, 0], route[:, 1], "-", color="#11a3a3", lw=3, alpha=0.6)
    ax.add_patch(Circle(route[0, :2], 0.25, color="#27ae60", alpha=0.5))
    ax.add_patch(Circle(route[-1, :2], 0.3, color="#3b82f6", alpha=0.4))
    ax.plot(head[:, 0], head[:, 1], "-", color="#555555", lw=1)
    ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_aspect("equal")
    ax.set_xlabel("forward along the segment [m, G1 scale]")
    bx.axhspan(*HAND_ZONE, color="#ff9f1a", alpha=0.08)
    bx.set_xlim(lo[0], hi[0]), bx.set_ylim(0, 1.5), bx.set_aspect("equal")
    bx.set_xlabel("forward [m]  (side view: objects within 1 m of the start line)"), bx.set_ylabel("z [m]")
    head_dot, = ax.plot([], [], "ks", ms=7)
    head_side, = bx.plot([], [], "ks", ms=6)
    top, side = ax.scatter([], [], s=30), bx.scatter([], [], s=24)
    title = ax.set_title("")

    frames = np.arange(t[0], t[-1], a.speed / a.fps)
    idx = np.searchsorted(t, frames).clip(0, len(t) - 1)

    def draw(k):
        i = idx[k]
        pts, cols = [], []
        for s_ in SIDES:
            if not z[f"{s_}_tracked"][i]:
                continue
            W = z[f"{s_}_wrist_xr"][i]
            E = z[f"{s_}_elbow_xr"][i] if f"{s_}_elbow_xr" in z else None   # native app: tracked elbow
            prox_scene = C.controller_proxy(Ti, W) if controllers else C.hand_proxy(Ti, W, E)
            prox_home = C.controller_proxy(Ti_home, W) if controllers else C.hand_proxy(Ti_home, W, E)
            for part in prox_scene:
                d, _ = obst.min_distance(*prox_scene[part])
                col = "#e8141d" if d < 0 else "#ff9f1a" if d < margin else "#3aa0ff"
                pts.append(prox_home[part][0])
                cols += [col] * len(prox_home[part][0])
        pts = np.vstack(pts) if pts else np.zeros((0, 3))
        top.set_offsets(pts[:, :2]), top.set_color(cols)
        side.set_offsets(pts[:, [0, 2]]), side.set_color(cols)
        head_dot.set_data([head[i, 0]], [head[i, 1]]), head_side.set_data([head[i, 0]], [head[i, 2]])
        cl = [z[f"clear_{s_}_{p_}"][i] for s_ in SIDES for p_ in ("hand", "forearm")]
        cl = [c for c in cl if np.isfinite(c)]
        title.set_text(f"{meta['scene_id']}  segment {seg['index'] + 1}  hazards {', '.join(h['id'] if isinstance(h, dict) else h for h in seg['bottlenecks']) or '-'}\n"
                       f"t={t[i]:5.1f}s  {meta['status']}  safe={meta['safe']}  "
                       f"hand/forearm clearance now {min(cl) if cl else float('nan'):+.3f} m")
        return top, side, head_dot, head_side, title

    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    out = a.out or os.path.join(a.take, "replay.mp4")
    anim = animation.FuncAnimation(fig, draw, frames=len(frames), blit=False)
    if out.endswith(".gif"):
        anim.save(out, writer=animation.PillowWriter(fps=a.fps))
    else:
        anim.save(out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=2500))
    print(out)


if __name__ == "__main__":
    main()
