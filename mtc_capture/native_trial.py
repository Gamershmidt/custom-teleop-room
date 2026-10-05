"""Trial for the native headset app (XRoboToolkit-MTC): legs, and wrists while looking ahead.

    source ~/Documents/teleop/env.sh
    python -m mtc_capture.tracker_trial --native            # 2 min, report -> native_trial.json
    python -m mtc_capture.tracker_trial --native --seconds 300 --record trial.npz

What to do in the headset: stand on the lobby grid, then walk around normally while looking ahead:
swing the arms, put the hands at the hips, behind the back, out to the sides, lift each foot.
In the headset, the 24 body joints are drawn (orange = legs, cyan = rest) and green spheres mark the
wrists computed from the controllers. Nothing goes into the dataset.

The report answers:
  - do the legs arrive?  body tracking coverage, foot / knee motion, PICO's tracking state
  - are the wrists tracked without looking at them?  controller coverage overall and with the
    controller outside the gaze cone (> 50 deg from where the head points)
  - one frame?  PICO's head joint vs the headset pose (should be < ~15 cm), feet near the floor
  - controller -> wrist offset: PICO's wrist joint in the controller frame (median), to pass as
    capture --wrist-offset
"""

import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np

from . import hud
from .g1_body import HUMAN_EYE_TO_HEIGHT
from .native_app import BODY
from .sources import CTRL_TO_WRIST_OFFSET, NativeSource

SIDES = ("left", "right")
LEGS = [n for n in BODY if any(w in n for w in ("hip", "knee", "ankle", "foot"))]
GAZE_CONE = math.radians(50)


def off_gaze(head, p):
    """Angle between where the head points (-z) and the direction to p."""
    d = p - head[:3, 3]
    n = np.linalg.norm(d)
    return math.acos(np.clip(np.dot(-head[:3, 2], d / n), -1, 1)) if n > 1e-6 else 0.0


def run(a):
    src = NativeSource(port=a.native_port, wrist_offset=a.wrist_offset)
    from .capture import _lan_ip
    print(__doc__.split("What to do")[0].strip().splitlines()[0])
    print(f"\nOpen 'XRoboToolkit Voice Beta' in the headset (Library > Unknown sources) (host {_lan_ip()}:{src.tv.port}, found by UDP beacon).\n"
          "Then walk around looking ahead: arms swinging, hands at the hips / behind the back / to the sides, "
          "lift each foot.\n")
    src.show(None, [])   # lobby grid: the floor level is visible too

    def keys():   # b + Enter: PICO's Motion Tracker calibration in the headset
        for line in sys.stdin:
            if line.strip()[:1] == "b":
                src.tv.command("body_calibrate")
                print("[body] opening the PICO Motion Tracker calibration in the headset")
    threading.Thread(target=keys, daemon=True).start()
    print("b + Enter: calibrate the Motion Trackers\n")
    t0 = time.monotonic()
    n = dict(frames=0, body=0, ctrl={h: 0 for h in SIDES}, away={h: 0 for h in SIDES},
             away_ok={h: 0 for h in SIDES})
    head_err, wrist_in_ctrl, foot_y, leg_y, rows = [], {h: [] for h in SIDES}, [], {k: [] for k in LEGS}, []
    eye_y = []
    last_n, hud_key, last_print, last_markers, state = -1, None, 0.0, 0.0, {}
    try:
        while time.monotonic() - t0 < a.seconds:
            now = time.monotonic()
            f, age, frames, _ = src.tv.latest()
            if frames == last_n:   # wait for the next headset frame
                time.sleep(0.002)
                continue
            last_n = frames
            s = src.read()
            state = ((f or {}).get("body") or {}).get("state") or {}
            if not s.head_valid:
                continue
            n["frames"] += 1
            eye_y.append(float(s.head[1, 3]))
            markers = []
            for h in SIDES:
                ctrl = s.extra[f"{h}_ctrl_xr"]
                if np.all(np.isfinite(ctrl)):
                    away = off_gaze(s.head, ctrl[:3, 3]) > GAZE_CONE
                    n["away"][h] += away
                    n["away_ok"][h] += away and s.tracked[h]
                n["ctrl"][h] += s.tracked[h]
                if s.tracked[h]:
                    markers.append(dict(tag="Sphere", key=f"wrist-{h}", args=[.035], position=s.wrist[h][:3, 3].tolist(),
                                        material=dict(color="#22c55e", emissive=True)))
            if s.body_age < 0.25 and len(s.body):
                n["body"] += 1
                J = s.body
                head_err.append(float(np.linalg.norm(J[BODY["head"], :3, 3] - s.head[:3, 3])))
                foot_y.append(float(min(J[BODY["left_foot"], 1, 3], J[BODY["right_foot"], 1, 3])))
                for k in LEGS:
                    leg_y[k].append(float(J[BODY[k], 1, 3]))
                for h in SIDES:
                    if s.tracked[h]:
                        C = s.extra[f"{h}_ctrl_xr"]
                        wrist_in_ctrl[h].append(C[:3, :3].T @ (J[BODY[f"{h}_wrist"], :3, 3] - C[:3, 3]))
                for name, j in BODY.items():
                    leg = name in LEGS
                    markers.append(dict(tag="Sphere", key=f"j-{name}", args=[.045 if leg else .03],
                                        position=J[j, :3, 3].tolist(),
                                        material=dict(color="#ff8c00" if leg else "#22c1e6", emissive=True)))
                if a.record:
                    rows.append(dict(t=now - t0, head=s.head.copy(), body=J.copy(),
                                     **{f"{h}_ctrl": s.extra[f"{h}_ctrl_xr"].copy() for h in SIDES}))
            if now - last_markers > 1 / 20:
                src.tv.xr_markers(markers)
                last_markers = now

            elapsed = now - t0
            fr = max(n["frames"], 1)
            body_cov = n["body"] / fr
            away_cov = {h: (n["away_ok"][h] / n["away"][h]) if n["away"][h] else None for h in SIDES}
            if body_cov > 0.5 and state.get("tracking"):
                title, color = "НОГИ И РУКИ ОТСЛЕЖИВАЮТСЯ", "green"
            elif not state.get("calibrated"):
                title, color = "ТРЕКЕРЫ НЕ ОТКАЛИБРОВАНЫ", "yellow"
            else:
                title, color = "ТЕЛО: " + str(state.get("text", "нет данных")).upper()[:28], "yellow"
            l2 = (f"тело {body_cov:.0%} · контроллеры L {n['ctrl']['left'] / fr:.0%} R {n['ctrl']['right'] / fr:.0%}"
                  + "".join(f" · вне взгляда {h[0].upper()} {v:.0%}" for h, v in away_cov.items() if v is not None))
            l3 = "Идите, глядя вперёд: руки вдоль тела, за спиной, в стороны; поднимите каждую ногу"
            key = (title, color, l2, int(elapsed))
            if key != hud_key:
                hud_key = key
                src.tv.hud(hud.render_lines(title, l2, l3, color), dict(distance=1.4, height=.17, aspect=4.0, below=.34))
            if now - last_print > 1.0:
                last_print = now
                flags = " ".join(f"{k}={state[k]}" for k in ("supported", "trackers", "body_mode", "calibrated", "started",
                                                             "start_result", "tracking") if k in state)
                print(f"[{elapsed:5.0f}s] {n['frames']} frames | body {body_cov:.0%} ({state.get('text', '-')}; {flags}) | "
                      f"ctrl L {n['ctrl']['left'] / fr:.0%} R {n['ctrl']['right'] / fr:.0%} | out of gaze "
                      + " ".join(f"{h[0].upper()} {v:.0%}" if v is not None else f"{h[0].upper()} -" for h, v in away_cov.items())
                      + (f" | head joint err {np.median(head_err[-200:]) * 100:.0f} cm" if head_err else ""))
            if not src.connected():
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        fr = max(n["frames"], 1)
        motion = {k: round(float(np.ptp(v)), 3) for k, v in leg_y.items() if len(v) > 10}
        feet_move = any(motion.get(k, 0) > 0.05 for k in ("left_foot", "right_foot", "left_ankle", "right_ankle"))
        offset = {h: np.median(np.array(v), axis=0).round(3).tolist() for h, v in wrist_in_ctrl.items() if len(v) > 20}
        away_cov = {h: round(n["away_ok"][h] / n["away"][h], 3) if n["away"][h] else None for h in SIDES}
        he = float(np.median(head_err)) if head_err else None
        eye = float(np.median(eye_y)) if eye_y else None
        problems = []
        if eye is not None and eye < 0.3:
            problems.append("the head pose stays at the floor: the app does not stream the headset pose")
        elif eye is not None and a.operator_height:
            want = HUMAN_EYE_TO_HEIGHT * a.operator_height
            if abs(eye - want) > 0.12:
                problems.append(f"eye height {eye:.2f} m, expected {want:.2f} m for {a.operator_height:.2f} m: "
                                "the headset floor is off; redo the PICO boundary / floor height")
        if n["frames"] == 0:
            problems.append("no frames from the headset app")
        if n["body"] / fr < 0.8:
            problems.append(f"body tracking only {n['body'] / fr:.0%} of frames ({state.get('text', '-')})")
        elif not feet_move:
            problems.append("feet barely move: walk and lift each foot, check the tracker straps")
        if he is not None and he > 0.15:
            problems.append(f"PICO head joint is {he * 100:.0f} cm from the headset: body joints are in another frame")
        for h, v in away_cov.items():
            if v is not None and v < 0.9:
                problems.append(f"{h} controller tracked only {v:.0%} of the time out of the gaze")
        summary = dict(
            verdict="OK: legs and wrists tracked while looking ahead" if not problems else "CHECK: " + "; ".join(problems),
            frames=n["frames"], seconds=round(time.monotonic() - t0, 1),
            body_coverage=round(n["body"] / fr, 3), body_state=state,
            controller_coverage={h: round(n["ctrl"][h] / fr, 3) for h in SIDES},
            controller_coverage_out_of_gaze=away_cov, out_of_gaze_fraction={h: round(n["away"][h] / fr, 3) for h in SIDES},
            leg_height_range_m=motion, lowest_foot_m=round(float(np.percentile(foot_y, 5)), 3) if foot_y else None,
            head_joint_error_m=round(he, 3) if he is not None else None,
            eye_height_m=round(eye, 3) if eye is not None else None, operator_height_m=a.operator_height,
            wrist_offset_in_controller_m=offset, wrist_offset_used=list(src.wrist_offset),
            hello=src.tv.hello())
        with open(a.report, "w") as fh:
            json.dump(summary, fh, indent=1)
        if a.record and rows:
            np.savez_compressed(a.record, **{k: np.array([r[k] for r in rows]) for k in rows[0]},
                                body_names=np.array(list(BODY)))
        print("\n" + "=" * 70 + f"\nVERDICT: {summary['verdict']}\n" + "=" * 70)
        print(f"body tracking {summary['body_coverage']:.0%} | controllers {summary['controller_coverage']} | "
              f"out of gaze {away_cov}")
        print(f"leg height ranges {motion}\nlowest foot {summary['lowest_foot_m']} m | head joint error {summary['head_joint_error_m']} m"
              f" | eye height {summary['eye_height_m']} m"
              + (f" (expected {HUMAN_EYE_TO_HEIGHT * a.operator_height:.2f} m)" if a.operator_height else ""))
        if offset:
            mean = np.mean([np.multiply(offset[h], [-1, 1, 1] if h == "left" else 1) for h in offset], axis=0)
            summary["suggested_wrist_offset"] = mean.round(3).tolist()
            with open(a.report, "w") as fh:
                json.dump(summary, fh, indent=1)
            print(f"wrist in the controller frame: {offset}  ->  capture --wrist-offset {mean[0]:.3f} {mean[1]:.3f} {mean[2]:.3f}"
                  " (right-hand frame, left mirrored in x)")
        print(f"report: {os.path.abspath(a.report)}" + (f"\nrecording: {os.path.abspath(a.record)}" if a.record and rows else ""))
        src.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--native-port", type=int, default=8013)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--report", default="native_trial.json")
    p.add_argument("--record", default=None, help="also save head, controllers and body joints (.npz)")
    p.add_argument("--operator-height", type=float, default=None, help="m; checks the headset floor via the eye height")
    p.add_argument("--wrist-offset", type=float, nargs=3, default=list(CTRL_TO_WRIST_OFFSET), metavar=("X", "Y", "Z"))
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
