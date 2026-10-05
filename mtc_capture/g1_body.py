"""Unitree G1 (29 DoF) + Dex3-1 dimensions used for scaling and hand clearance.

Measured from g1_29dof_with_hand.urdf / meshes (teleop/sim/assets/g1) at the zero pose,
pelvis frame, with the pelvis at its standing height:

    pelvis -> ankle_roll 0.757 m, sole ~0.03 m below  -> pelvis height ~0.79 m
    head_link mesh top  0.531 m above pelvis           -> overall height ~1.32 m
    d435 (head camera)  0.474 m above pelvis           -> eye height ~1.26 m
    elbow joint -> wrist_yaw joint 0.184 m               (forearm)
    wrist_yaw joint -> index/middle fingertip 0.217 m    (Dex3 hand, open)
    finger links ~26 mm thick, palm 88 mm wide, forearm links ~60 mm thick

On this branch the Dex3 fingers are fixed in a stand pose, and the collision model uses
one padded box per hand (data/assets/unitree_g1/dex3/geometry-contract.json, in the
wrist_yaw frame). The clearance proxy uses that envelope: it reaches ~0.22 m past the
wrist and is ~10 cm thick, whereas a human hand scaled to G1 size is only ~0.14 m long.
"""

import json
import os

G1_HEIGHT = 1.32          # m, floor to top of head
G1_EYE_HEIGHT = 1.264     # m, floor to d435 head camera
G1_PELVIS_HEIGHT = 0.79   # m
G1_SHOULDER_HEIGHT = 1.08  # m, shoulder roll joint

DEX3_REACH = 0.217        # m, wrist_yaw joint -> fingertip, open hand
DEX3_FINGER_RADIUS = 0.013
DEX3_PALM_HALF_WIDTH = 0.044
FOREARM_LENGTH = 0.184    # m, elbow joint -> wrist_yaw joint
FOREARM_RADIUS = 0.032

TORSO_RADIUS = 0.16       # shoulders incl. shoulder links (half width ~0.14 + link radius)
HEAD_RADIUS = 0.10
BODY_RADIUS_2D = 0.20     # footprint radius used for the traversability check

# Height band where the hands and forearms of an upright G1 move while walking.
HAND_ZONE = (0.45, 1.20)

# Mean eye height / stature for adults; used only if the operator height is not given.
HUMAN_EYE_TO_HEIGHT = 0.936


def _dex3_envelope():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "assets", "unitree_g1", "dex3",
                        "geometry-contract.json")
    try:
        with open(path) as f:
            env = json.load(f)["envelopes"]["left"]
        c, h = env["center"], env["half_size"]
        return c[0] - h[0], c[0] + h[0], max(h[1], h[2]), "geometry-contract.json"
    except (OSError, KeyError, ValueError):
        return 0.036, 0.222, 0.052, "built-in (contract not found)"


# along the hand axis from the wrist_yaw joint [m], cross-section radius [m], source
DEX3_ENVELOPE_START, DEX3_ENVELOPE_END, DEX3_ENVELOPE_RADIUS, DEX3_ENVELOPE_SOURCE = _dex3_envelope()
