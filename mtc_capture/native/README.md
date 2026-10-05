# XRoboToolkit-MTC: native headset app for capture (legs + wrists while looking ahead)

The browser capture (`vr_app.py`, WebXR) can't record the legs or the arms reliably:

- The PICO browser gives web pages no Motion Tracker data. Every browser take so far has
  `body_joint_names: []`.
- WebXR hand tracking is camera-based, so it stops when the hands leave the view (76-98 % coverage,
  `tracking_lost` takes).

The native app is the [XRoboToolkit Unity client](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client)
(MIT) with an MTC layer added (`unity/Assets/MtcCapture/`). An app gets what the PICO SDK offers:

| | browser (`capture`) | native app (`capture --native`) |
|---|---|---|
| legs | none | PICO body tracking: 24 joints from the headset, both controllers and 2 ankle Motion Trackers |
| wrists | camera hand tracking, lost out of view | **controllers held in the hands**: LEDs seen by the side and lower cameras, IMU through gaps |
| elbows | not tracked (forearm guessed from the wrist axis) | from body tracking; the forearm proxy points at the tracked elbow |
| fingers | 25 joints | none (the Dex3 fingers are fixed, so the clearance proxy doesn't use them) |
| room, panel, sounds | three.js in the page | drawn by the app from the same capture messages |

The Dex3 clearance envelope is a capsule symmetric about the finger axis, so a held controller gives
everything it needs: the wrist position and the finger direction.

```
PICO 4 Ultra: XRoboToolkit-MTC app                         capture computer (Mac or Ubuntu)
  head, controllers, 24 body joints  --ws://host:8013/mtc-->  native_app.NativeLink -> capture.py -> takes/
  room / status panel / sound cues  <---------------------    (UDP beacon on 8014 announces the host)
```

No XRoboToolkit PC service is needed for capture: the app talks to `capture.py` directly. The
app's normal XRoboToolkit functions still work.

## Build the app (once, and after changes under `unity/`)

1. Clone the client and install the layer:
   ```bash
   git clone https://github.com/XR-Robotics/XRoboToolkit-Unity-Client.git ~/src/XRoboToolkit-Unity-Client
   mtc_capture/native/install_into_fork.sh ~/src/XRoboToolkit-Unity-Client
   ```
   This copies `Assets/MtcCapture/` (5 scripts and a shader) and adds the
   `CHANGE_WIFI_MULTICAST_STATE` permission so Android delivers the capture's UDP beacon. The layer
   starts by itself (`RuntimeInitializeOnLoadMethod`), so no scene or prefab edits are needed.
2. Open the project in **Unity 2022.3.16f1** (the version XRoboToolkit requires) with Android
   Build Support, set the platform to Android, and build an APK (File > Build Settings > Build).
   The client's README covers its own build details.
3. Install it on the headset (developer mode on, USB):
   ```bash
   adb install -r XRoboToolkit-MTC.apk
   ```

The layer hasn't been compiled here yet (no Unity on this Mac). If the first build reports
errors, they will be in `Assets/MtcCapture/`; send them back.

## Host address

The app finds the capture computer by its UDP beacon (`MTC_CAPTURE 8013` to port 8014, once a second).
If the network drops broadcasts, put the address in a file on the headset:

```bash
echo "192.168.1.20:8013" > mtc_host.txt
adb push mtc_host.txt /sdcard/Android/data/<app id>/files/mtc_host.txt
```

The app id is `com.xrobotoolkit.client` for a build from the Unity menu, and
`com.xrobotoolkit.client.voicebeta` for XRoboToolkit's batch build
(`-executeMethod VoiceDuplexBetaBuilder.BuildBatch`, Unity's debug key, no keystore). That one
installs next to a regular XRoboToolkit app and shows up as "XRoboToolkit Voice Beta".

Until it connects, the app shows a status text ahead of you: what it is looking for, and the
body-tracking state. While it is connected, XRoboToolkit's own panels are hidden.

## Motion Trackers

Pair the two trackers and strap them to the lower legs. The app starts PICO full-body tracking
(`BTM_FULL_BODY_HIGH`) as soon as the trackers are calibrated. To calibrate, type **`b` + Enter**
in the capture or trial terminal, which opens PICO's calibration in the headset, or use the PICO
Motion Tracker app. The app restarts body tracking by itself if it is lost for more than 5 s.

## Check it: the trial (2 minutes, nothing goes into the dataset)

```bash
source ~/Documents/teleop/env.sh && cd ~/Documents/mtc-samr/Click-and-Traverse
python -m mtc_capture.tracker_trial --native            # report -> native_trial.json
```

Walk around while looking ahead: arms swinging, hands at the hips, behind the back, out to the
sides; lift each foot. In the headset the 24 joints are drawn (orange = legs) along with green
wrist spheres. The verdict checks:

- body tracking coverage and that the feet move;
- controller coverage when the controller is more than 50° from the gaze (the "not looking at
  the hands" case);
- that PICO's head joint lies within ~15 cm of the headset pose, i.e. body joints and head are in
  one frame;
- the controller → wrist offset, measured against PICO's wrist joint. Pass the printed
  `--wrist-offset X Y Z` to capture. The default is 8 cm behind the controller origin.

## Capture

```bash
mtc_capture/run_capture.sh alice 1.74 --native [--wrist-offset X Y Z]
# or: python -m mtc_capture.capture --native --operator alice --operator-height 1.74 --scenes .../manifest.json
```

Everything else is unchanged: scenes, segments, the pads, hazard colours, the status panel,
sounds, the spectator view (`spectator.py`) and the take format. The differences:

- **Calibrate** by holding both triggers + both grips for 1 s, standing upright on the home spot.
  `c` / `C` on the keyboard still work.
- **Takes** have `input: "native"`, `body_joints_xr` (N, 24, 4, 4) with `body_joint_names`
  `pico:<joint>`, raw `{left,right}_ctrl_xr`, `{left,right}_elbow_xr`, and `headset_t` (headset
  clock, unix ns). `{side}_wrist_xr` is in the usual WebXR wrist-joint convention, so `replay`,
  `view_take` and the retargeting read it as they read hand tracking. `meta.native` records the
  wrist offset and the app's hello (model, OS, tracking origin).

## How touches are detected

Every tracking sample, capture puts the **G1's collision shapes** at the operator and measures their
distance to the room's boxes (clearance.py; numbers in G1 metres; at your height multiply by
1/alpha = height/1.32, about 1.39 for 1.84 m):

| part | shape | from |
|---|---|---|
| hand | Dex3 envelope: capsule 0.037-0.222 m past the wrist along the fingers, radius 0.052 | controller -> wrist (`--wrist-offset`), fingers = where the controller points |
| forearm | capsule 0.184 m from the wrist, radius 0.032 | towards PICO's tracked elbow |
| body | head sphere 0.10; torso column radius 0.115 from shoulders to pelvis; shoulder spheres 0.055 at +-0.10 m sideways (no upper arms: PICO only guesses the elbows) | head, PICO's chest (spine3) and shoulder line (turns with you) |

A distance below 0 is a touch; below `--hand-margin` (5 cm) the object turns orange.

Why touches can surprise: the shapes are the **robot's**, scaled to you. The Dex3 envelope reaches
about 31 cm past your wrist with a 7 cm radius at 1.84 m, longer and fatter than your hand, and it
points where the controller points. Until 2026-10-05 the body was a round 16 cm column under the
head that did not turn with you, so sideways passages registered "body" touches the real robot
would not have (10 of 14 takes in the narrow rooms). Choose the **robot** self view (right B
button) to see exactly these shapes, coloured by their clearance: blue, orange, red.

## Tracking quality: glitches, live skeleton, POV video

**Controller glitches.** When the headset cameras lose a controller's LEDs (behind the body, low at
the side, fast head turns), PICO keeps reporting it as tracked while it extrapolates: the pose jumps
at 20-110 m/s, up to 1.4 m from the head, and snaps back. On the replay these show up as "clouds"
of hand motion. Capture (sources.GlitchFilter) rejects a sample that jumps faster than 6 m/s since
the last good one, is more than 1.15 m from the eyes, or is more than 0.30 m from PICO's own wrist
joint. The hand stays untracked until 0.1 s after the last bad sample, which means no clearance
check, no false touches, and the "hand lost" cue. Takes store `{side}_glitch` per sample and
`meta.controller_glitch_fraction`; glitchy takes fall below the 90 % tracking needed for `safe`.

**Self view in the headset** (right controller **B** or `v` + Enter cycles; `--self-view` sets the start):
`skeleton` PICO's body skeleton (orange legs, cyan arms and spine; no head or neck, so nothing in front
of the eyes) and both controller wrists with a stick along the fingers, green when tracked, red during
a glitch; `robot` the collision shapes above, coloured by clearance; `both`; `off`.

**Third-person panel** (left controller **Y** or `t` + Enter): a small panel at the upper left of the
view showing you from 1.8 m behind and 0.6 m above (room + self view), 20 fps.

**POV video.** The app renders the operator's view off-screen (room, route, status panel, skeleton;
not the passthrough camera image, which apps never get) at 640x480, 15 fps and streams it as JPEG.
Every take saves `pov.mp4` and `pov.npz` (`t`: take clock as in motion.npz; `headset_t`: headset
clock, as in motion.npz `headset_t`).

**Replay** (`view_take`): the G1 skeleton (links as lines) next to the meshes, the PICO skeleton,
glitch frames ("CONTROLLER GLITCH" in the readout; arms hold their last good pose; intervals in the
take info), and a "POV (headset view)" panel showing the frame at the slider's time. Takes recorded
before these changes get their glitches detected on load.

## Protocol (`native_app.py`)

Frame: OpenXR basis (x right, y up, z back), metres, floor-level origin; poses `[x, y, z, qx, qy, qz, qw]`.

- app → host
  - `hello`: `{app, version, device, model, os, origin}`
  - `track`, every frame: `{t, origin, head, head_ok, ctrl: {left, right: {ok, pose, trigger, grip, primary, secondary, menu, axis}}, body: {state: {supported, calibrated, started, tracking, text}, joints: [24 poses] | null}}`
- app → host, binary: `P` + headset time (8 bytes, little-endian unix ns) + JPEG: one POV frame
- host → app
  - `scene` `{lobby, els}` and `xr` `{els}`: each element is
    `{k: key, s: box|cylinder|sphere|plane, d: dims, m: row-major 4x4 in the XR frame, c: colour, o: opacity, e: emissive}`.
    The capture applies its anchor before sending; the app only converts to Unity (`S M S`, `S = diag(1, 1, -1)`).
  - `hud` `{png: base64, layout}`, `event` `{name}`, `cmd` `{name: body_calibrate | body_start}`

`python -m mtc_capture.native_fake_headset` plays the app (walks each segment, holds triggers to
calibrate) to test capture without a headset.

## Not yet verified on the headset

- The C# has been checked against the SDK sources in the client, not compiled.
- Whether PICO's body-tracking `localPose` shares the app's floor-level tracking space. The trial's
  head-joint check catches a mismatch.
- The default controller → wrist rotation: the back of the hand towards the controller's outer
  side. Clearance doesn't depend on it, but retargeting the wrist roll does. The raw controller
  poses are in every take, so it can be refit offline.
