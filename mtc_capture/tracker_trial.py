"""Explicit trial: do the PICO Motion Trackers (legs) reach the capture page, and what do they give?

    source ~/Documents/teleop/env.sh
    python -m mtc_capture.tracker_trial                     # 2 min, report -> tracker_trial.json
    python -m mtc_capture.tracker_trial --seconds 300 --record legs.npz

Serves the capture page (same as a session: lobby, hands, body/tracker probe from xr_body.js) and
  - draws every body joint / tracker the browser reports as a marker in VR (orange = leg joints or
    trackers below 0.6 m, cyan = others), so you can see whether your feet are tracked;
  - shows a live status panel in the headset and a live report in this terminal: enabled XR
    features, input sources, body joint names, sample rate, foot heights and how much they move;
  - writes a report with a verdict. With --record, also the raw body / tracker stream (.npz).
Nothing is recorded into the dataset.
"""

import argparse
import os as _os
_os.environ.setdefault("AIOHTTP_NOSENDFILE", "1")
import json
import math
import os
import time

import numpy as np

from . import hud
from .sources import XRSource

CHECKLIST = """
Motion tracker trial
====================
Before starting:
  1. PICO: pair both Motion Trackers (Settings > Motion Tracker / the PICO Motion Tracker app) and
     strap them to the ankles as PICO shows.
  2. Calibrate them in the Motion Tracker app (stand upright, then look down at the feet until
     both trackers are recognised). Recalibrate whenever a strap moves.
  3. Keep hand tracking on; put the controllers down.
Then:
  4. In the PICO browser open  {url}
     press 'Virtual Reality' and allow any tracking permission the browser asks for.
  5. Stand on the lobby grid, look down at your feet, then walk a few steps and lift each foot.
     Orange markers at your feet = leg data reaches the capture. The panel below your view and
     this terminal report what arrives.
"""

LEG_WORDS = ("leg", "knee", "foot", "ankle", "toe", "hip")


def is_leg(name, pos):
    return any(w in name.lower() for w in LEG_WORDS) or (name.startswith("input:") and pos[1] < 0.6)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8012)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--report", default="tracker_trial.json")
    p.add_argument("--record", default=None, help="also save the raw body / tracker stream (.npz)")
    p.add_argument("--native", action="store_true",
                   help="test the native headset app instead of the browser (legs + wrists while looking ahead)")
    a, rest = p.parse_known_args()
    if a.native:   # native_trial has its own options (--native-port, --wrist-offset)
        from . import native_trial
        args = ["--seconds", str(a.seconds)] + (["--record", a.record] if a.record else []) + rest
        if a.report != "tracker_trial.json":
            args += ["--report", a.report]
        return native_trial.main(args)
    if rest:
        p.error(f"unrecognized arguments: {' '.join(rest)}")

    src = XRSource(port=a.port)
    from .capture import _lan_ip
    ip = _lan_ip()
    print(CHECKLIST.format(url=f"https://{ip}:{a.port}/?ws=wss://{ip}:{a.port}"))
    src.show(None, [])   # lobby grid, so the floor level is visible too

    t0 = time.monotonic()
    names_seen, sample_times, heights, rows = set(), set(), {}, []
    first_body_t, hud_key, last_print, last_markers = None, None, 0.0, 0.0
    try:
        while time.monotonic() - t0 < a.seconds:
            now = time.monotonic()
            s = src.read()
            fresh = s.body_age < 0.25 and len(s.body_names) > 0
            markers = []
            if fresh:
                first_body_t = first_body_t or now
                sample_times.add(round(now - s.body_age, 4))
                for name, P in zip(s.body_names, s.body):
                    names_seen.add(name)
                    pos = P[:3, 3]
                    if not np.all(np.isfinite(pos)):
                        continue
                    leg = is_leg(name, pos)
                    heights.setdefault(name, []).append(float(pos[1]))
                    markers.append(dict(tag="Sphere", key=f"trk-{len(markers)}", args=[.05 if leg else .03, 16, 12],
                                        position=pos.tolist(), materialType="standard",
                                        material=dict(color="#ff8c00" if leg else "#22c1e6", emissive="#ff8c00" if leg else "#22c1e6",
                                                      emissiveIntensity=.6)))
                if a.record:
                    rows.append((now - t0, dict(zip(s.body_names, s.body.copy()))))
            if now - last_markers > 1 / 20:   # <= 20 updates/s to the headset
                src.tv.xr_markers(markers)
                last_markers = now

            legs = sorted(n for n in names_seen if heights.get(n) and is_leg(n, [0, np.median(heights[n]), 0]))
            rate = len(sample_times) / (now - first_body_t) if first_body_t and now - first_body_t > 1.0 else 0.0
            foot = {n: heights[n][-1] for n in legs if heights.get(n)}
            moving = {n: float(np.ptp(heights[n][-300:])) for n in legs if len(heights.get(n, [])) > 5}
            elapsed = now - t0
            if legs:
                title, color = "ТРЕКЕРЫ: НОГИ ВИДНЫ", "green"
                l3 = "Пройдитесь и поднимите каждую ногу — маркеры должны двигаться"
            elif names_seen:
                title, color = "ЕСТЬ ДАННЫЕ ТЕЛА, НО НЕ НОГИ", "yellow"
                l3 = "Посмотрите вниз на ступни; проверьте калибровку трекеров"
            elif not s.head_valid:
                title, color = "ОЖИДАНИЕ ОЧКОВ", "grey"
                l3 = "Нажмите Virtual Reality, положите контроллеры, покажите руки"
            elif elapsed < 25:
                title, color = "ПОИСК ТРЕКЕРОВ…", "grey"
                l3 = "Посмотрите вниз на ступни и пройдитесь"
            else:
                title, color = "ТРЕКЕРЫ НЕ ПЕРЕДАЮТСЯ", "red"
                l3 = "Браузер PICO не даёт данные трекеров странице"
            l2 = (f"{len(names_seen)} точек · {rate:.0f} Гц · " +
                  (" ".join(f"{n.split(':')[-1][:14]} {h:.2f}м" for n, h in list(foot.items())[:3]) or "ступни не найдены"))
            key = (title, color, l2[:40], int(elapsed))
            if key != hud_key:
                hud_key = key
                src.tv.hud(hud.render_lines(title, l2, l3, color), dict(distance=1.4, height=.17, aspect=4.0, below=.34))
            if now - last_print > 1.0:
                last_print = now
                rep = src.probe_report()
                srcs = rep.get("inputSources", [])
                print(f"[{elapsed:5.0f}s] head {'ok' if s.head_valid else '--'} | hands L:{'ok' if s.tracked.get('left') else '--'} "
                      f"R:{'ok' if s.tracked.get('right') else '--'} | features {rep.get('enabledFeatures')} | "
                      f"input sources {len(srcs)} {[ (x.get('handedness'), (x.get('profiles') or ['?'])[0], 'hand' if x.get('hand') else '') for x in srcs]} | "
                      f"body/tracker points {len(names_seen)} @ {rate:.0f} Hz | legs {len(legs)}"
                      + (f" | foot heights {', '.join(f'{n}={h:.2f}' for n, h in foot.items())}" if foot else ""))
    except KeyboardInterrupt:
        pass
    finally:
        rep = src.probe_report()
        legs = sorted(n for n in names_seen if heights.get(n) and is_leg(n, [0, np.median(heights[n]), 0]))
        dur = (time.monotonic() - first_body_t) if first_body_t else 0.0
        summary = dict(
            verdict=("LEGS AVAILABLE in the browser: capture records them in every take (body_joints_xr)" if legs else
                     "BODY DATA BUT NO LEG JOINTS: check tracker pairing/calibration and repeat" if names_seen else
                     "NOT EXPOSED: the PICO browser gives web pages no body / tracker data -> use the XRoboToolkit route"),
            body_points=sorted(names_seen), leg_points=legs,
            sample_rate_hz=round(len(sample_times) / dur, 1) if dur > 0 else 0.0,
            leg_height_range_m={n: [round(min(heights[n]), 3), round(max(heights[n]), 3)] for n in legs},
            enabled_features=rep.get("enabledFeatures"), input_sources=rep.get("inputSources"),
            requested=rep.get("requested"), body_api=rep.get("bodyApi"), probe_error=rep.get("error"),
            user_agent=src.headset_user_agent or rep.get("userAgent"), device=src.device,
            xr_frame_members=rep.get("xrFrameMembers"), xr_session_members=rep.get("xrSessionMembers"))
        with open(a.report, "w") as f:
            json.dump(summary, f, indent=1)
        if a.record and rows:
            names = sorted({n for _, r in rows for n in r})
            arr = np.full((len(rows), len(names), 4, 4), np.nan)
            for i, (_, r) in enumerate(rows):
                for j, n in enumerate(names):
                    if n in r:
                        arr[i, j] = r[n]
            np.savez_compressed(a.record, t=np.array([t for t, _ in rows]), body_xr=arr, names=np.array(names))
        print("\n" + "=" * 70 + f"\nVERDICT: {summary['verdict']}\n" + "=" * 70)
        print(f"points: {summary['body_points'] or 'none'}\nlegs: {legs or 'none'}   rate: {summary['sample_rate_hz']} Hz")
        for n, r in summary["leg_height_range_m"].items():
            print(f"  {n}: height {r[0]:.2f}..{r[1]:.2f} m" + ("  (moves: tracked)" if r[1] - r[0] > .05 else "  (static?)"))
        print(f"report: {os.path.abspath(a.report)}" + (f"\nrecording: {os.path.abspath(a.record)}" if a.record and rows else ""))
        src.close()


if __name__ == "__main__":
    main()
