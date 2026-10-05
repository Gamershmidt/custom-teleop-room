# Capture session checklist (Pico 4 / Meta Quest 3, hands_v2)

Two people: the **operator** wears the headset; the **assistant** runs the Mac and watches the space.

## Once per headset: Pico 4

1. Pico browser trusts the local CA: install `~/.config/xr_teleoperate/rootCA.pem` (copy it as
   `rootCA.crt`) as a CA certificate; see `~/Documents/televuer/README.md`, "Connect a Pico headset".
   Never copy `rootCA.key` or `key.pem`.
2. Pico hand tracking on (Settings > Interaction > Hand tracking).

## Pico Motion Trackers (legs)

1. Pair the trackers and put them on (ankles, as PICO instructs), then **calibrate them in the
   PICO Motion Tracker app** before opening the browser. Recalibrate whenever the straps move.
2. **Run the tracker trial once** before collecting (2 minutes, nothing goes into the dataset):
   `python -m mtc_capture.tracker_trial` (optionally `--record legs.npz`). In the headset, orange
   markers appear at your feet if leg data arrives, and the panel below your view says
   ТРЕКЕРЫ: НОГИ ВИДНЫ / ЕСТЬ ДАННЫЕ ТЕЛА, НО НЕ НОГИ / ТРЕКЕРЫ НЕ ПЕРЕДАЮТСЯ. Walk and lift each foot.
   The terminal ends with a VERDICT and writes `tracker_trial.json`.
3. The capture page asks the browser for WebXR body tracking and records every body joint or extra
   tracker the browser exposes (`body_joints_xr` in each take). **Whether the Pico browser passes
   the Motion Trackers to web pages is not certain**: the preflight headset test shows it.
   - `body / leg data: N joints ... legs: ...` means the legs are recorded.
   - `body / leg data: none` means this browser does not expose them. Head and hands are still recorded.
     Leg data then needs a native PICO app; send `xr_probe.json` (written by the test) to the team.

## Native app (PICO 4 Ultra): legs + wrists without looking at them

Use this instead of the browser when recording legs: the PICO browser does not pass the Motion
Trackers to web pages. Build and install the app once ([native/README.md](native/README.md)), then:

1. Strap the Motion Trackers to the lower legs. **Hold both controllers** throughout; they
   track the wrists wherever the arms are.
2. Start **XRoboToolkit Voice Beta** in the headset. It finds the computer by itself; until then a
   status text ahead of you says what it is waiting for.
3. Trackers not calibrated (the status says so): type `b` + Enter in the terminal, then follow
   PICO's calibration in the headset.
4. First time, and whenever something looks off: `python -m mtc_capture.tracker_trial --native`
   (2 min). Walk looking ahead, arms at the hips, behind the back, to the sides; lift each foot.
   It should end with `VERDICT: OK`, and prints a `--wrist-offset` to use.
5. Session: `mtc_capture/run_capture.sh alice 1.74 --native [--wrist-offset X Y Z]`.
   **Calibrate by holding both triggers + both grips for 1 s**, upright on the home spot. The pads,
   colours, panel and sounds are the same as in the browser.

## Once per headset: Meta Quest 3 / 3S

1. **Hand tracking on**, with automatic switching between hands and controllers: Settings >
   Movement tracking > Hand and body tracking (menu names change between Quest OS versions).
2. **Boundary: roomscale, not stationary.** Draw it around the whole capture area (Settings >
   Physical space > Boundary) and check the floor level there. A stationary boundary moves the
   origin with you and breaks the room placement.
3. **Certificate:** the Meta Quest Browser shows a warning for the Mac's self-signed certificate.
   Choose **Advanced > Proceed**. This is needed once per address (IP), and again after the
   certificate is re-issued.
4. **When entering VR**, the browser asks for permission to use hand tracking: **Allow**.
5. **Optional, USB instead of Wi-Fi** (no Wi-Fi dropouts; the Mac's IP no longer matters):
   turn on developer mode (Meta Horizon phone app > Devices > Headset settings > Developer mode),
   `brew install --cask android-platform-tools`, plug in a long data USB-C cable, then run
   `mtc_capture/quest_usb.sh` and open `https://localhost:8012/?ws=wss://localhost:8012`. The cable
   has to reach the whole capture area.

On the Quest, looking at your palm while pinching opens the system menu. Do the calibration
gesture with the palms facing away from you (backs of the hands towards your face). If the menu opens, close it
and go back to the page.

## Room setup

- Free floor of **5.5 m × 3.0 m** (or pass `--space FWD WIDTH` with your real size; long routes
  are then cut shorter). Mark the **home spot** with tape at the middle of one short side; the
  operator always starts there facing down the long axis.
- Draw the headset boundary around the whole area and set the **floor height** carefully. A wrong
  floor scales everything; preflight checks it, and the lobby grid should be at your feet.
- Mac and headset on the same Wi-Fi (or a Quest on USB), no guest/client isolation. Good light for hand tracking.

## Start (assistant)

```bash
cd ~/Documents/mtc-samr/Click-and-Traverse
source ~/Documents/teleop/env.sh
python -m mtc_capture.preflight --operator alice --operator-height 1.74 --headset 30   # first session of the day
mtc_capture/run_capture.sh alice 1.74          # preflight + capture; resumes where the last session stopped
```

Chaotic rooms have several routes (the terminal shows `route 1 (3 routes)`); each is walked in turn,
and only the current one is drawn as the teal strip. Add `--reverse` to also walk each route backwards.

Height in metres, barefoot/shoes as worn in the session. `--per-operator` makes every operator
walk the whole set (recommended with 3 operators).

Headset browser (Pico browser or Meta Quest Browser): open the URL printed by the script and press
**Virtual Reality**. The headset model is detected from the browser and saved in every take (`meta.json` `device`).

## Spectator (second person)

A live MuJoCo view of the real G1 + Dex3 moving through the current room, on the Mac (a second
screen works well). Start it in a second terminal while the capture runs:

```bash
source ~/Documents/teleop/env.sh && cd ~/Documents/mtc-samr/Click-and-Traverse
mjpython -m mtc_capture.spectator                         # same Mac as the capture
mjpython -m mtc_capture.spectator --host <capture-mac-ip>  # another computer with this repo + data
```

**Operator POV on any laptop or phone:** run `python -m mtc_capture.spectator --serve` on the
capture Mac and open `http://<mac-ip>:8120` in a browser on the same network (no install). It shows
the operator's point of view, rebuilt from the headset tracking at ~20 fps: the room, the route, the
hazard colours, their tracked hands, and the same status panel they see. A small chase view of the
G1 is underneath. The window and the browser stream can run at the same time.

The window shows the room, the current route segment and pads, furniture turning orange or red as in the
headset, the robot (pelvis under the headset, arms solved from the operator's wrists, Dex3 fixed,
legs in a standing pose), orange markers for leg trackers, and a status panel: state, room,
route/segment, takes, hand tracking, legs. The camera follows the robot; drag to orbit, scroll to zoom.
It only reads the capture's state, so it can be started and closed at any time.

## What the operator sees and hears in the headset

A status panel hangs just below the centre of view (Russian by default, `--hud-lang en` for English;
`--hud-offset` / `--hud-height` move and resize it):

| panel | colour | meaning |
|---|---|---|
| НЕ ОТКАЛИБРОВАНО | grey | pinch both hands together in front of the face to place the room |
| КАЛИБРОВКА ОТКЛОНЕНА | red | eye height does not match the operator height: stand upright or fix the headset floor |
| ВЕРНИТЕСЬ НА СТАРТ | grey | walk to the grey start pad, face the blue goal |
| ПОВЕРНИТЕСЬ К ЦЕЛИ | yellow | on the pad but facing away; turn to the blue goal |
| ПРИГОТОВЬТЕСЬ… 3/2/1 | yellow | countdown, stand still |
| ● ЗАПИСЬ N с | green | recording; orange when a hand is out of the cameras' view, red after a contact (lists what was touched) |
| ГОТОВО: БЕЗОПАСНО / БЫЛО КАСАНИЕ / ПРЕРВАНО | blue / orange / grey | result of the take, then back to the start |
| ВСЁ ЗАПИСАНО | blue | every selected segment is done |

The second line always shows route · segment · take.

Sounds, synthesised in the headset browser: a chime when the room is placed, a low double tone when
calibration is refused, ticks for the countdown, a rising tone when recording starts, a soft click
when a hand comes within 5 cm of something, a buzz on contact, a double beep when a hand drops out
of view, a rising arpeggio for a clean take, a falling tone for a take with contact, a descending
tone for an aborted take. The browser unlocks sound with the "Virtual Reality" tap; keep the
headset volume up.

## Operator

1. On the home spot, stand upright, face down the floor, and **pinch with both hands, held together
   in front of your face**, for 1 s: the room appears at G1 scale. If the terminal says
   `[calib] REFUSED`, the headset floor is off (or you weren't upright): fix the floor height in the
   boundary setup; the lobby grid must be at your feet.
2. Stay on the **start pad**, face the **blue goal pad**: yellow, filling ring, then **green** =
   recording.
3. Walk upright along the teal strip to the blue pad at a normal, careful pace. **Protect the
   hands**: tuck, raise over table tops, turn sideways - whatever a person would do, without ducking.
   Keep the hands roughly in front of the body (the headset only tracks what it sees).
4. Orange = the robot's hand/forearm is within 5 cm of something; red = touched. The beacon above
   the goal turns red when the take is no longer clean: finish it anyway or abort.
5. At the goal the take ends. Walk back to the home spot; the next segment appears.

Abort a take: the same gesture, both hands pinched together in front of the face (or assistant
`x`). Re-place the room: same gesture while not recording.

## Assistant keys (type + Enter)

`x` abort · `C` calibrate despite the floor check · `d` discard the last take (moved to `takes_discarded/`) · `n` skip segment ·
`N` skip scene · `c` recalibrate · `s` status · `q` quit.

Watch the terminal: `[take] ... tracked L 97% R 94% ... SAFE / not safe`. Low tracking = hands out
of view. Discard takes where the operator stumbled, stopped to think, or walked through furniture
on purpose.

## Breaks and end

Every ~20-30 min: a break (VR fatigue changes how people move). `q` or Ctrl-C ends the session;
nothing recorded is lost. Progress: `python -m mtc_capture.status`.

Data: `data/mtc_capture/takes/` (+ `scenes/`, `scene_sets/`). Back it up after each session.
