"""Pre-session checks for headset capture (Pico 4 / Meta Quest 3). Run in the teleop environment:

    source ~/Documents/teleop/env.sh
    python -m mtc_capture.preflight --operator alice --operator-height 1.74            # Mac side
    python -m mtc_capture.preflight --operator alice --operator-height 1.74 --headset 30  # + live headset test

Mac side: Python packages, TLS certificate valid for this Mac's current IP and signed by the CA the
headset trusts, port free, scene set readable, disk space, this operator's progress.
--headset N: serves the capture page, waits for the Pico, then measures for N s: head and hand
sample rates, how often each hand is tracked, the head height against the operator's height (a
wrong floor level shows up here), and the tracked hand size. Nothing is recorded.
--fix-cert re-issues the certificate for the current IP with ~/Documents/teleop/scripts/make_certs.sh
(same CA, so the headset keeps trusting it).
"""

import argparse
import os as _os
_os.environ.setdefault("AIOHTTP_NOSENDFILE", "1")
import json
import os
import shutil
import socket
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
TELEOP = os.path.expanduser("~/Documents/teleop")
CA = os.path.expanduser("~/.config/xr_teleoperate/rootCA.pem")
DEFAULT_SET = os.path.join(REPO, "data", "mtc_capture", "scene_sets", "messy_v3", "manifest.json")

OK, WARN, FAIL = "\033[32mOK\033[0m", "\033[33mWARN\033[0m", "\033[31mFAIL\033[0m"
results = []


def report(status, what, detail=""):
    results.append(status)
    print(f"  [{status}] {what}" + (f": {detail}" if detail else ""))


def lan_ips():
    ips = []
    if shutil.which("ipconfig") and sys.platform == "darwin":
        for ifc in ("en0", "en1", "en2", "en3"):
            ip = subprocess.run(["ipconfig", "getifaddr", ifc], capture_output=True, text=True).stdout.strip()
            if ip:
                ips.append(ip)
    else:   # Linux: IPv4 addresses of the interfaces that are up, without loopback, docker and VPN tunnels
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "up"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            f = line.split()
            if len(f) > 3 and not f[1].startswith(("lo", "docker", "br-", "veth", "tailscale", "tun", "wg")):
                ips.append(f[3].split("/")[0])
    return ips


def check_packages():
    print("Python environment")
    for mod in ("numpy", "vuer", "televuer", "aiohttp"):
        try:
            __import__(mod)
            report(OK, mod)
        except Exception as e:
            report(FAIL, mod, f"{e} (use `source ~/Documents/teleop/env.sh`)")
    try:
        from televuer.televuer import TeleVuer  # noqa: F401
        from . import capture  # noqa: F401
        report(OK, "mtc_capture imports")
    except Exception as e:
        report(FAIL, "mtc_capture imports", repr(e))


def check_cert(fix):
    print("TLS certificate (the Pico browser needs HTTPS with this Mac's IP in the certificate)")
    cert, key = os.environ.get("XR_TELEOP_CERT"), os.environ.get("XR_TELEOP_KEY")
    if not (cert and key and os.path.exists(cert) and os.path.exists(key)):
        report(FAIL, "certificate", "XR_TELEOP_CERT/KEY not set: `source ~/Documents/teleop/env.sh`")
        return
    text = subprocess.run(["openssl", "x509", "-in", cert, "-noout", "-text"], capture_output=True, text=True).stdout
    ips = lan_ips()
    if not ips:
        report(FAIL, "network", "no Wi-Fi/Ethernet IP: connect the Mac to the headset's network")
        return
    missing = [ip for ip in ips if f"IP Address:{ip}" not in text]
    if missing and fix:
        old = [t.split(":")[1].strip().rstrip(",") for t in text.split() if t.startswith("Address:")]
        subprocess.run([os.path.join(TELEOP, "scripts", "make_certs.sh"), *old], check=True, cwd=TELEOP)
        text = subprocess.run(["openssl", "x509", "-in", cert, "-noout", "-text"], capture_output=True, text=True).stdout
        missing = [ip for ip in ips if f"IP Address:{ip}" not in text]
    if missing:
        report(FAIL, "certificate covers this Mac's IP", f"{', '.join(missing)} missing: rerun with --fix-cert")
    else:
        report(OK, "certificate covers this Mac's IP", ", ".join(ips))
    v = subprocess.run(["openssl", "verify", "-CAfile", CA, cert], capture_output=True, text=True)
    report(OK if v.returncode == 0 else FAIL, "signed by the local CA", CA if v.returncode == 0 else v.stdout + v.stderr)
    end = subprocess.run(["openssl", "x509", "-in", cert, "-noout", "-checkend", str(14 * 86400)], capture_output=True)
    report(OK if end.returncode == 0 else WARN, "valid for 14+ days")


def check_port(port):
    print("Network")
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # ignore TIME_WAIT leftovers of the last session
    try:
        s.bind(("0.0.0.0", port))
        report(OK, f"port {port} free")
    except OSError:
        report(FAIL, f"port {port} free", f"in use: `lsof -nP -iTCP:{port} -sTCP:LISTEN`, or use --port")
    finally:
        s.close()
    ips = lan_ips()
    if ips:
        report(OK, "headset URL (Wi-Fi)", f"https://{ips[0]}:{port}/?ws=wss://{ips[0]}:{port}")
    report(OK, "headset URL (Quest over USB, after mtc_capture/quest_usb.sh)", f"https://localhost:{port}/?ws=wss://localhost:{port}")


def check_native_port(port):
    """Native headset app: a plain WebSocket on `port` (no TLS) and the UDP beacon on port + 1."""
    print("Network (native headset app)")
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", port))
        report(OK, f"port {port} free")
    except OSError:
        report(FAIL, f"port {port} free", f"in use (a capture or trial still running?): `lsof -nP -iTCP:{port} -sTCP:LISTEN`")
    finally:
        s.close()
    ips = lan_ips()
    report(OK if ips else WARN, "host address for the app", f"{ips[0]}:{port} (sent by UDP beacon)" if ips else "no network address found")


def check_scenes(a):
    print("Scene set and progress")
    from . import furniture
    from .capture import G1_HEIGHT, _existing_takes, segment_key
    if not os.path.exists(a.scenes):
        report(FAIL, "scene set", f"{a.scenes} missing: build it with make_scene_set.py (branch .venv)")
        return
    with open(a.scenes) as f:
        m = json.load(f)
    base = os.path.dirname(os.path.abspath(a.scenes))
    alpha = G1_HEIGHT / a.operator_height
    takes_dir = os.path.join(os.path.abspath(a.out), "takes")
    done = _existing_takes(takes_dir, a.operator if a.per_operator else None, False)
    want = got = segs = segs_done = 0
    widest = (0.0, 0.0)
    by_e = {e["path"]: e for e in m["scenes"]}
    for rel in m["capture_order"]:
        e = by_e[rel]
        if e["split"] not in ("train", "validation"):
            continue
        sc = furniture.load(os.path.join(base, rel))
        for seg in sc.all_segments(3.0, "auto", (alpha, *a.space), reverse=a.reverse):
            n = e["takes_per_segment"]
            have = min(n, done[segment_key(sc.scene_id, seg.s0, seg.s1, seg.route_id)])
            want, got, segs, segs_done = want + n, got + have, segs + 1, segs_done + (have >= n)
            fw = seg.view.floor_need(seg.s0, seg.s1, alpha)
            widest = (max(widest[0], fw[0]), max(widest[1], fw[1]))
    report(OK, "scene set", f"{m['name']}: {len(m['scenes'])} scenes, {segs} train+validation segments at alpha {alpha:.3f}")
    report(OK if widest[0] <= a.space[0] + .05 and widest[1] <= a.space[1] + .05 else WARN, "floor",
           f"largest segment needs {widest[0]:.1f} x {widest[1]:.1f} m; free floor must be {a.space[0]} x {a.space[1]} m "
           "ahead of / centred on the home spot, inside the Pico boundary")
    hours = (want - got) / 60.0
    report(OK, "progress", f"{got}/{want} successful takes, {segs_done}/{segs} segments complete, "
                           f"~{hours:.1f} h of capture left at ~60 takes/h")
    free = shutil.disk_usage(os.path.abspath(a.out) if os.path.exists(a.out) else REPO).free / 1e9
    need = (want - got) * 0.6 / 1e3
    report(OK if free > 5 * need + 5 else WARN, "disk", f"{free:.0f} GB free, ~{need:.1f} GB needed")


HINTS = {   # menu paths change between OS versions; these are where they usually are
    "quest": dict(
        hands="put the controllers down (the Quest switches to hands after a few seconds; enable the automatic "
              "switch under Settings > Movement tracking > Hand and body tracking)",
        floor="redo the floor in Settings > Physical space > Boundary (use a roomscale boundary, not stationary)",
        page="reload the page in the Meta Quest Browser and press 'Virtual Reality' again"),
    "pico": dict(
        hands="put the controllers down or switch them off; Settings > Interaction > Hand tracking must be on",
        floor="redo the floor height in the Pico boundary setup",
        page="reload the page in the Pico browser and press 'Virtual Reality' again"),
}


def hints(device):
    return HINTS["quest" if device.startswith("quest") else "pico" if device == "pico" else "pico"] \
        if device != "unknown" else {k: f"{HINTS['quest'][k]} / Pico: {HINTS['pico'][k]}" for k in HINTS["quest"]}


LEG_WORDS = ("leg", "knee", "foot", "ankle", "toe", "hip", "tracker", "motion")


def body_report(src, a, dur):
    """What the headset browser exposes beyond head and hands (xr_body.js), and whether legs arrive."""
    rep = src.probe_report()
    feats = rep.get("enabledFeatures")
    report(OK, "XR features enabled", f"{feats}")
    extra = [x for x in rep.get("inputSources", []) if not x.get("hand") and not (
        x.get("handedness") in ("left", "right") and x.get("mode") == "tracked-pointer")]
    report(OK, "input sources", f"{len(rep.get('inputSources', []))} total, {len(extra)} besides hands/controllers"
           + (f": {extra}" if extra else ""))
    bj = rep.get("bodyJoints")
    names = sorted(src.body_names_seen)
    legs = [n for n in names if any(w in n.lower() for w in LEG_WORDS)]
    rate = len(src.body_sample_times) / max(dur, 1e-6)
    if names:
        report(OK if legs else WARN, "body / leg data", f"{len(names)} joints at ~{rate:.0f} Hz"
               + (f"; legs: {', '.join(legs[:8])}{' ...' if len(legs) > 8 else ''}" if legs else "; no leg joints among them"))
    else:
        report(WARN, "body / leg data", "none: this browser exposes no body tracking or extra trackers to web pages "
               f"(XRFrame.body {'present' if bj is not None else 'absent'}, XRBody API "
               f"{'yes' if (rep.get('bodyApi') or {}).get('XRBody') else 'no'}). Legs need a native app (PICO SDK); "
               "head and hands are still recorded")
    if rep.get("error"):
        report(WARN, "body probe", str(rep["error"])[:200])
    if a.probe_out:
        with open(a.probe_out, "w") as f:
            json.dump(rep, f, indent=1)
        report(OK, "full browser report", a.probe_out)


def check_headset(a):
    print(f"Headset (live, {a.headset:.0f} s; nothing is recorded)")
    from .sources import XRSource
    src = XRSource(port=a.port)
    ip = (lan_ips() or ["127.0.0.1"])[0]
    print(f"  Open https://{ip}:{a.port}/?ws=wss://{ip}:{a.port} in the headset browser (Pico browser or Meta Quest\n"
          "  Browser; on a certificate warning: Advanced > Proceed), press 'Virtual Reality'.\n"
          "  You should see a floor grid with a green ring and four posts. Put the controllers down,\n"
          "  stand upright and move both hands in front of you.")
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < 180:
            s = src.read()
            if s.head_valid:
                break
            time.sleep(.1)
        else:
            report(FAIL, "headset connected", "no head pose from a hand-tracking headset in 3 min: page open in the headset "
                   "browser and in VR? controllers put down and hands in view? (Wi-Fi, certificate?)")
            return
        dev = src.device
        report(OK, "headset connected", f"{dev} ({src.headset_user_agent[:90] or 'no User-Agent'})")
        hint = hints(dev)
        heads, head_changes, tracked, hand_t, sizes, last = [], 0, {"left": 0, "right": 0}, {"left": set(), "right": set()}, [], None
        src.body_sample_times, src.body_names_seen = set(), set()
        raw = {"left": 0, "right": 0}
        n, t1 = 0, time.monotonic()
        while time.monotonic() - t1 < a.headset:
            s = src.read()
            n += 1
            if last is None or not np.array_equal(s.head, last):
                head_changes += 1
                last = s.head.copy()
            heads.append(s.head[1, 3])
            if s.body_age < 0.25 and s.body_names:
                src.body_sample_times.add(round(time.monotonic() - s.body_age, 4))   # distinct samples, not polls
                src.body_names_seen.update(s.body_names)
            for h in ("left", "right"):
                if np.any(s.wrist[h][:3, :3]) and np.any(s.joints[h]):
                    raw[h] += 1
                if s.tracked[h]:
                    tracked[h] += 1
                    hand_t[h].add(s.sample_t[h])
                    kp = np.asarray(s.joints[h])
                    sizes.append(np.linalg.norm(kp[14] - kp[0]))
            time.sleep(1 / 240)
        dur = time.monotonic() - t1
        rate = head_changes / dur
        report(OK if rate > 45 else WARN, "head pose rate", f"{rate:.0f} Hz" + ("" if rate > 45 else
               f": is the page in VR (not the 2D page)? {hint['page']}"))
        for h in ("left", "right"):
            frac, hz = tracked[h] / n, len(hand_t[h]) / dur
            if raw[h] == 0:
                why = f": no hand data at all. {hint['hands']}; hold the hands in front of you"
            elif frac < .8:
                why = ": keep the hands in front of the headset (it only tracks what its cameras see)"
            else:
                why = ""
            report(OK if frac > .8 and hz > 25 else WARN, f"{h} hand", f"tracked {frac:.0%} of the time, {hz:.0f} Hz" + why)
        body_report(src, a, dur)
        eye = float(np.median(heads))
        expect = .936 * a.operator_height
        report(OK if abs(eye - expect) < .12 else WARN, "floor level",
               f"head {eye:.2f} m above the floor, expected ~{expect:.2f} m for {a.operator_height} m"
               + ("" if abs(eye - expect) < .12 else ": stand upright (not sitting); look down, the grid should be at "
                  f"your feet; if not, {hint['floor']}"))
        if sizes:
            sz = float(np.median(sizes))
            report(OK if .14 < sz < .23 else WARN, "hand size (wrist -> middle tip)", f"{sz * 100:.0f} cm")
    finally:
        src.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--operator", required=True)
    p.add_argument("--operator-height", type=float, required=True)
    p.add_argument("--scenes", default=DEFAULT_SET)
    p.add_argument("--out", default=os.path.join(REPO, "data", "mtc_capture"))
    p.add_argument("--space", type=float, nargs=2, default=[5.5, 3.0])
    p.add_argument("--per-operator", action="store_true")
    p.add_argument("--reverse", action="store_true", help="count reversed routes too (as capture --reverse)")
    p.add_argument("--port", type=int, default=8012)
    p.add_argument("--headset", type=float, default=0, help="live headset test for N seconds")
    p.add_argument("--fix-cert", action="store_true")
    p.add_argument("--probe-out", default="xr_probe.json", help="where to save the browser's full XR capability report")
    p.add_argument("--native", action="store_true", help="native headset app: no TLS certificate, port --native-port")
    p.add_argument("--native-port", type=int, default=8013)
    a = p.parse_args()
    check_packages()
    if a.native:   # the browser's TLS certificate and port 8012 are not used
        check_native_port(a.native_port)
    else:
        check_cert(a.fix_cert)
        check_port(a.port)
    check_scenes(a)
    if a.headset and FAIL not in results:
        check_headset(a)
    bad = results.count(FAIL)
    print(f"\n{'READY' if not bad else 'NOT READY'}: {bad} failed, {results.count(WARN)} warnings")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
