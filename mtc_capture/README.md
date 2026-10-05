# mtc_capture: VR demonstrations in the Click-and-Traverse furniture scenes

This is the data-collection stage of MTC (MTC-Capturer, Sec. III-B of the paper), applied to this
branch's whole-body / hand-protection task. An operator in a Pico 4 or Meta Quest 3 headset walks the route of a
**Click-and-Traverse furniture scene** (`cat-furniture-scene-v1`). The scenes are the same
tables, chairs, rows, gates and hand passages the policy trains on. The room is shown at G1
scale, so the operator moves as the robot has to: upright, with the hands kept clear of the
furniture. The headset's tracking is recorded raw, and retargeting happens later.

```
Pico (WebXR hands) --wss:8012--> televuer/Vuer on the Mac --> capture loop --> takes/<id>/motion.npz
       ^                                                              |
       '------- scene.json boxes, drawn in the headset (three.js) <---'
```

## Scenes

Scenes come from this branch's generators, or from any `scene.json` they wrote:

| `--generate` | generator | notes |
|---|---|---|
| `dense[:family]` (default `dense:table`) | `cat_ppo.furniture.scenes` | 9 tables, 36 chairs, closed row ends, 6 gates. Families: `table`, `chair`, `slalom`, `asymmetric`, `overhead`, `turn`, `mixed`. `mixed` and `overhead` include 1.2 m beams that need ducking. |
| `pilot[:family]`, `open` | same | single table encounter / empty room |
| `random[:furniture\|generic_clutter]` | `random_rooms` | packed at random, with an admitted route |
| `clutter[:family]` | `clutter` | the dense topology with crates, partitions and shelves |
| `hand_table_aisle[:easy\|medium\|hard]`, `hand_shelf_passage[:…]` | `hand_passages` | height-selective arm passages. The certificate is skipped without scikit-fmm; the geometry is unchanged. |
| `--scenes DIR…` | anything | e.g. table-edge and contrastive passages, which need the branch's JAX environment: generate them there, then pass the directory |

**Segments.** A dense route is ~40 m, which no tracked floor can hold at 1/alpha. Capture
therefore works on route windows. By default (`--segments auto`) each window is centred on a
bottleneck gate, `--segment-length` 3 m long at G1 scale, which is about 4 m of real floor.
Routes without gates are cut into consecutive pieces, and routes up to 3.75 m stay whole. Every
segment is anchored so that its start is on the operator's **home spot**. The operator walks the
segment, then walks back home for the next one. Each `[segment]` log line prints the real floor
the segment needs; with `--space FWD WIDTH` it warns when a segment needs more.

## The hands_v2 scene set

Only the medium / protected variant of each branch family is kept, plus seven chaotic patterns
(`chaotic_rooms.PATTERNS`): medium, hard, storage (crate towers, pallets, shelves, plank piles),
office (desks heaped with monitors, chairs everywhere, open cabinets), classroom (tables in skewed
rows, chairs upside down on them or toppled), corridor (8.5-10 x 2.6-3.2 m, junk along both walls)
and debris (plank piles, poles, angled partitions, toppled chairs).

`make_scene_set.py` builds a balanced, certified set from the generators above. Run it in the
branch environment (`.venv`), because it needs JAX and scikit-fmm:

```bash
source .venv/bin/activate && source .env
python -m mtc_capture.make_scene_set --name hands_v2          # about 2 min
python -m mtc_capture.make_scene_set --name pilot --scale 0.3 # a small pilot
```

| family / variant (train) | scenes | segments | takes (3 per segment) |
|---|---:|---:|---:|
| hand_table_aisle medium | 10 | 10 | 30 |
| hand_shelf_passage medium | 10 | 10 | 30 |
| table_edge forward_protected | 10 | 30 | 90 |
| contrastive narrow | 8 | 24 | 72 |
| dense table | 4 | 24 | 72 |
| random furniture | 8 | 25 | 75 |
| random generic_clutter | 8 | 35 | 105 |
| chaotic medium | 12 | 26 | 78 |
| chaotic hard | 12 | 26 | 78 |
| chaotic storage | 12 | 26 | 78 |
| chaotic office | 12 | 27 | 81 |
| chaotic classroom | 12 | 26 | 78 |
| chaotic corridor | 12 | 35 | 105 |
| chaotic debris | 12 | 25 | 75 |
| **train total** | **142** | **349** | **1047** |

Besides train there are 24 validation scenes (57 segments, 1 take each) and 27 test scenes
(69 segments) for evaluating policies in simulation; test scenes are not captured. Segments are
centred on each hazard: bottleneck gates, or the hand-contrast zones of the table-edge and
contrastive passages. `manifest.json` records every scene and its planned takes. Its
`capture_order` cycles through the families, so a collection stopped early is still balanced.
`overview.png` shows one example per family.

```bash
python -m mtc_capture.capture --operator alice --operator-height 1.74 \
    --scenes data/mtc_capture/scene_sets/hands_v2/manifest.json
```

**Chaotic rooms** (`chaotic_rooms.py`) are messy rooms built for hand protection: tables heaped with
boxes, laptops, monitors and planks overhanging their edges; chairs tossed, toppled, or upside down
on a table with the backrest hanging over; twisted crate towers, planks bridging crates, poles
across chairs; shelves with things sticking out; cabinets with the door open and a drawer out;
partitions at angles. Routes come from the branch's own admission (`_admit_routes`), planned so
that an upright G1 never walks under or over anything (0.02-1.40 m). A final tier then puts
hand-height objects along both route sides, near face 0.30-0.42 m from the route centre. Each
scene's `hand_hazards` records how much of the route has hand-height clutter within 0.45 m, and
whether it met the target (medium >= 45 %, hard >= 60 %).

**Several routes per chaotic room.** Each chaotic room reserves four start/goal pockets near the walls
(corridor: its two ends) and plans distinct routes between them: across, diagonal, and alternatives
between the same pockets (found by blocking the previous path). Routes overlapping another by more
than 55 % are dropped. Annealing thins the clutter until the room has at least 3 routes (corridor 1),
then the hand-height tier re-lines every route, keeping all of them clear. `scene.json` lists them in
`routes` (case index, name like `W-E alt`, length, hazard coverage); `start_goals` holds each route
forwards and reversed. Capture walks every listed route (take `meta.json` `segment.route_id`);
`--reverse` also walks every route backwards, which gives corridors their second direction.

**Seeing the scenes.** `overview.png` has a top view of one scene per family. For 3D, use the branch's
Viser viewer: native G1 meshes at the start, Dex3 hand envelopes, route, walls, and camera buttons.

```bash
source .venv/bin/activate && source .env
python -m mtc_capture.view_scenes prepare                 # one bundle per family/variant (--all, --family dense)
.visual-venv/bin/python -m mtc_capture.view_scenes serve  # prints http://127.0.0.1:8085 ... one port per scene
```

Capture resumes by itself: segments that already have their successful takes in `takes/` are
skipped. With `--per-operator`, only this operator's takes count, so each operator can walk the
whole set.

## Collection sessions

Operator checklist: [OPERATOR.md](OPERATOR.md). In short:

```bash
source ~/Documents/teleop/env.sh
python -m mtc_capture.preflight --operator alice --operator-height 1.74 --headset 30  # Mac + live headset checks
mtc_capture/run_capture.sh alice 1.74 [--space 5.5 3.0] [--per-operator]           # a session; resumes
python -m mtc_capture.status                                                        # progress per family
```

`preflight` checks the environment, the TLS certificate against the Mac's current IP (and
re-issues it with `--fix-cert`, same CA), the port, the scene set, disk space and progress. With
`--headset N` it also measures head/hand tracking rates and coverage, the floor level against
the operator's height, and the tracked hand size, recording nothing. Segments are cut to fit the
free floor (`--space`, default 5.5 × 3.0 m from the home spot), starting on the home spot facing
their goal. `--operator-height` is required: alpha = 1.32 / height, as in the paper. Synthetic
(`--fake`) runs go to `data/mtc_capture/_fake_demo/`, never into the dataset.

## Run

Uses the `teleop` environment and the `televuer` fork from `~/Documents/teleop` (plus `colorlog`,
which `cat_ppo` imports). The Pico TLS setup is the one described in `~/Documents/teleop/README.md`.

```bash
cd ~/Documents/mtc-samr/Click-and-Traverse          # on feature/whole-body-furniture-traversal
source ~/Documents/teleop/env.sh

python -m mtc_capture.capture --operator alice --operator-height 1.74 --generate dense:table --space 5.5 3
python -m mtc_capture.capture --operator alice --operator-height 1.74 --generate hand_table_aisle:hard
python -m mtc_capture.capture --operator alice --operator-height 1.74 --scenes generated_scenes/

# no headset
python -m mtc_capture.capture --operator test --operator-height 1.75 --fake careless --view   # https://localhost:8012
python -m mtc_capture.replay data/mtc_capture/takes/<take_id>            # -> replay.mp4
```

### Operator protocol

1. **Calibrate once.** Stand at one end of the free floor, facing along it, upright. Pinch both
   hands, held together in front of your face, for 1 s (or the assistant types `c`). alpha is
   1.32 / `--operator-height`. Calibration is refused if the eye height doesn't match the operator
   height within 15 cm (wrong headset floor, or not standing upright); `C` forces it.
2. **Start pad.** The segment's start pad is at your feet, and the teal strip on the floor is the
   route. Face along it; the pad turns yellow, fills, then turns green: recording.
3. **Follow the route to the blue pad.** Furniture turns **orange** when the G1 hand envelope or
   forearm is within 5 cm, and **red** on contact. The beacon above the goal turns red once a take
   is not clean.
4. **Walk back home.** The next segment appears, faint until you step on the pad. Hands up +
   pinch while recording aborts a take.

Assistant keys (+ Enter): `c` calibrate · `x` abort · `d` move the last take to
`takes_discarded/` · `n` next segment · `N` next scene · `s` status · `q` quit.

## Status panel and sounds in the headset

`hud.py` renders the operator's status panel on the Mac as a PNG (system font, so Cyrillic works
offline). `vr_app.py` shows it head-locked, 1 m ahead and `--hud-offset` below the gaze, re-sent only
when the content changes. `xr_feedback.js` (injected into the page) plays WebAudio cues for events
pushed over `wss://…/mtc/events`: calibrated, refused, arming, tick, start, warn, touch, hand_lost,
success, unsafe, abort, new_segment. `--hud-lang ru|en` sets the panel language.

## Spectator view

`capture.py` publishes its live state over ZMQ (`--spectator-port`, default 5591; 0 turns it off).
`spectator.py` shows the Unitree G1 + Dex3 model (teleop's `g1_29dof_with_hand.xml`) posed from it,
in MuJoCo:
- pelvis under the headset at the G1 head-camera offset, heading = gaze yaw
- arms from the teleop G1 arm IK, using the operator's wrists in televuer's conventions, at G1 scale
- Dex3 in the branch's fixed stand pose, legs nominal

`--serve [PORT]` (default 8120) instead streams two MJPEG views to any browser: the operator's point
of view (a 90° camera at the headset pose, robot hidden, tracked hand joints, and the same
status-panel PNG as in the headset) and the chase view. It is a reconstruction from tracking, not a
screen capture of the headset.

It also draws the room boxes with the capture's hazard colours, the current segment, the pads and
any leg-tracker joints. `mjpython -m mtc_capture.spectator` opens a window;
`python -m mtc_capture.spectator --offscreen view.mp4 --seconds 30` records a video instead. It is
a live preview only; the dataset stays raw.

## Take format

`takes/<scene_id>__seg<k>__<operator>__<YYYYmmdd-HHMMSS>/`

**`motion.npz`**: one row per new tracking sample (head pose changed or a new XR hand sample):

| key | shape | meaning |
|---|---|---|
| `t`, `t_unix` | (N,) | host monotonic s since the take started; wall-clock time |
| `head_xr` | (N,4,4) | headset pose |
| `head_new` | (N,) | head pose changed since the previous row |
| `{left,right}_wrist_xr` | (N,4,4) | WebXR wrist joint (+Z to the elbow, fingers along −Z) |
| `{left,right}_joints_xr`, `…_joint_rot_xr` | (N,25,3), (N,25,3,3) | the 25 WebXR hand joints (`meta.joint_names`) |
| `{left,right}_sample_t`, `…_tracked` | (N,) | XR sample time (repeated = no new sample, nan = invalid); tracked within 0.25 s |
| `{left,right}_{pinch,pinchValue,squeeze,squeezeValue}` | (N,) | WebXR hand state |
| `clear_{left,right}_{hand,forearm}`, `clear_body` | (N,) | clearance of the G1 proxy [m, G1 scale], negative = penetration |
| `near_*` | (N,) | nearest object: key into `scene.labels` (furniture_id, or the box name for walls) |

With `--controllers`, the joint, pinch and squeeze keys are replaced by trigger, squeeze,
button and thumbstick values.

All `*_xr` values are raw: WebXR local-floor, OpenXR basis (y up), human scale. To go to the
**scene.json frame** (room corner origin, z up, G1 scale):

```python
T = np.array(meta["T_scene_from_xr"])      # similarity: rotation, translation, scale alpha
p_scene = p_xr @ T[:3, :3].T + T[:3, 3]
R_scene = (T[:3, :3] / meta["alpha"]) @ R_xr
```

**`meta.json`** holds (among others `device`, e.g. `quest3` / `pico`, detected from the headset browser, and `headset_user_agent`):
- `status` (success / aborted / timeout / tracking_lost), `safe` and `hands_safe`
- `scene_id`, `scene_file` (a copy in `scenes/`), `scene_geometry_hash` and `scene_source`
- `segment`: route window, start pose, goal and the gates it contains
- alpha and the operator's height; `T_xr_from_scene`, `T_scene_from_xr` and `T_xr_from_home`
- tracking coverage, the clearance summary, the objects touched and by which proxy part
- the proxy dimensions and the frame conventions

`safe` = success, no contact by the hands, forearms or body, and both hands tracked for at least
90 % of the take. Every take is kept, so filtering is up to you.

## Body / leg tracking (Pico Motion Trackers)

`xr_body.js` is injected into the capture page next to televuer's hand bridge. It asks for WebXR
`body-tracking` as an optional feature (the session starts either way). Each XR frame it reads
`XRFrame.body` joints and any input source that is neither a hand nor a controller (a tracker).
These go to the Mac over `wss://…/mtc/body`, in the same XR reference space as the head and
hands. Once a second it posts a capability report to `/mtc/probe`: enabled features, input
sources, body joint names and XR API members. `preflight --headset` prints that report, saves it
to `xr_probe.json`, and says whether leg joints arrive.

Takes then contain `body_joints_xr` (N, J, 4, 4; NaN where a joint was not reported) and
`body_tracked` (N,), and `meta.json` lists `body_joint_names` and `body_tracking_coverage`. Joint
names are the browser's own, prefixed `body:` (WebXR body joints) or `input:<i>:<profile>`
(extra tracked input sources). Whether the Pico browser exposes the Motion Trackers to web pages
is decided by the browser, not by this code; if the probe shows none, leg data needs a native
PICO SDK app.

## Clearance proxy (live, G1 scale)

- **Hand:** this branch's Dex3 collision envelope (`data/assets/unitree_g1/dex3/geometry-contract.json`):
  a capsule running 0.037–0.222 m past the wrist along the tracked hand axis, 5.2 cm radius. The
  robot's fingers are fixed, so the operator's finger pose does not change it; raw finger joints
  are recorded all the same.
- **Forearm:** a 32 mm capsule 0.184 m from the wrist towards the elbow. The elbow is not tracked.
- **Body:** a head sphere and a torso capsule under the head. Legs are not tracked.

This proxy is a feedback signal and a first filter. It does not replace checking the retargeted
robot against the 35 approved collision primitives.

## Native headset app: legs + wrists while looking ahead (`--native`)

The PICO browser gives web pages no Motion Tracker data (every browser take so far has no body
joints), and its hand tracking stops when the hands leave the view. `capture --native` uses a
native PICO app instead: the XRoboToolkit Unity client with an MTC layer
([native/README.md](native/README.md)). It records PICO body tracking (24 joints, legs from two
ankle Motion Trackers) and the wrists from controllers held in the hands, which stay tracked
while the operator looks ahead. Elbows from body tracking orient the forearm proxy. The room,
status panel and sounds are drawn by the app from the same capture messages.

```bash
python -m mtc_capture.tracker_trial --native        # check legs + wrists out of view first
mtc_capture/run_capture.sh alice 1.74 --native      # calibrate: both triggers + grips for 1 s
```

## Known limits

- **Browser: hands out of view are not tracked.** Pico hand tracking only sees hands in front of
  the headset. `--controllers` tracks further out but records no fingers. `--native` fixes both
  this and the legs.
- **Browser: legs only if the browser exposes them**, and the PICO browser does not (see "Body /
  leg tracking"). Use `--native`.
- **Long routes are split into segments.** Takes are cut at segment boundaries, and each one
  starts from a standstill.
- **Not yet used on a headset.** Tested with a scripted WebSocket client that plays the headset,
  and with `--fake`.
