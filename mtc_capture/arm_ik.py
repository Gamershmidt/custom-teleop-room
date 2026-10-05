"""Retarget a take onto the branch's G1 (29 DoF + fixed Dex3) for viewing: the base follows the
operator's head, both 7-DoF arms follow the tracked wrists by IK, the legs walk with planned
footsteps (gait.py), and the real robot collision geoms are checked against the furniture.
Runs in the branch environment (MuJoCo).

Wrist target: the tracked OpenXR wrist joint (scene frame, G1 scale) becomes the pose of the
G1 wrist_yaw_link, with televuer's OpenXR -> Unitree arm convention (same as teleoperation).
Base: placed so the G1 head site is at the tracked head (x, y and height), yaw from the smoothed
gaze. Legs: with PICO body tracking (native app takes), each G1 ankle goes to the operator's
tracked ankle (x, y at G1 scale; lift above that ankle's planted height; heading from ankle to
toes). Without it, footsteps are planned along the base path (gait.py). Each 6-DoF leg reaches
its foot pose by IK (knees bend to the head height). The waist stays at its default.

This is a kinematic preview, not a physically consistent retarget: no balance, no contacts,
no self-collision; IK residuals are recorded per frame.
"""

import os

import numpy as np

from .gait import plan_footsteps

SIDES = ("left", "right")
LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
# televuer tv_wrapper: OpenXR wrist axes -> Unitree wrist_yaw_link axes (right-multiplied)
R_OPENXR_ROBOT = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], float)
R_TO_UNITREE_ARM = {"left": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float),
                    "right": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)}
# collision groups reported per side (branch geoms from assemble_scene_xml)
GROUPS = {"hand": "furniture_{s}_hand_envelope", "forearm": "furniture_{s}_forearm",
          "upper_arm": ("furniture_{s}_upper_arm", "furniture_{s}_shoulder")}
BODY_GEOMS = ("furniture_torso", "furniture_head", "furniture_pelvis")
HIGHLIGHT = {"ok": (40, 140, 255), "warn": (255, 150, 0), "contact": (235, 40, 40)}


def build_model(scene_dir):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import mujoco
    from cat_ppo.envs.g1 import constants
    from cat_ppo.envs.g1.env_furniture import assemble_scene_xml
    from cat_ppo.furniture.scenes import load_scene
    model = mujoco.MjModel.from_xml_string(assemble_scene_xml(load_scene(scene_dir)))
    return model, np.asarray(constants.DEFAULT_QPOS, float)


def export_links(model, out_dir):
    """One GLB per robot body (visual meshes in the body frame) + links.json."""
    import json
    import mujoco
    import trimesh
    os.makedirs(out_dir, exist_ok=True)
    links, envelopes, hands = [], {}, []
    for b in range(1, model.nbody):
        scene = trimesh.Scene()
        for g in np.flatnonzero(model.geom_bodyid == b):
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_contype[g]:
                continue
            k = model.geom_dataid[g]
            va, vn, fa, fn = model.mesh_vertadr[k], model.mesh_vertnum[k], model.mesh_faceadr[k], model.mesh_facenum[k]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, model.geom_quat[g])
            v = model.mesh_vert[va:va + vn] @ R.reshape(3, 3).T + model.geom_pos[g]
            mesh = trimesh.Trimesh(vertices=v, faces=model.mesh_face[fa:fa + fn], process=False)
            mesh.visual.face_colors = np.clip(model.geom_rgba[g] * 255, 0, 255).astype(np.uint8)
            scene.add_geometry(mesh, node_name=f"geom_{g}")
        if scene.geometry:
            name = model.body(b).name
            scene.export(os.path.join(out_dir, f"{name}.glb"))
            links.append(name)
            if any(name == f"{s}_wrist_yaw_link" or name.startswith(f"{s}_hand_") for s in SIDES):
                for state, rgb in HIGHLIGHT.items():          # Dex3 hand in its clearance colour
                    for geom in scene.geometry.values():
                        geom.visual.face_colors = np.r_[rgb, 255].astype(np.uint8)
                    scene.export(os.path.join(out_dir, f"{name}__{state}.glb"))
                hands.append(name)
    kinds = {mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
             mujoco.mjtGeom.mjGEOM_SPHERE: "sphere"}
    for s in SIDES:
        for group, names in GROUPS.items():
            for x in names if isinstance(names, tuple) else (names,):
                g = model.geom(x.format(s=s)).id
                envelopes[x.format(s=s)] = dict(group=f"{s}_{group}", body=model.body(model.geom_bodyid[g]).name,
                                                type=kinds[model.geom_type[g]], pos=model.geom_pos[g].tolist(),
                                                quat=model.geom_quat[g].tolist(), size=model.geom_size[g].tolist())
    with open(os.path.join(out_dir, "links.json"), "w") as f:
        json.dump(dict(links=links, hand_links=hands, collision=envelopes), f, indent=1)
    return links


def _crouch_table(model, data, q_default):
    """Knee bend a -> pelvis drop keeping the soles flat on the floor: hip -a, knee +2a, ankle -a."""
    import mujoco
    idx = {j: [model.jnt_qposadr[model.joint(f"{s}_{j}_joint").id] for s in SIDES]
           for j in ("hip_pitch", "knee", "ankle_pitch")}
    foot = model.site("left_foot").id
    knee_lo, knee_hi = model.jnt_range[model.joint("left_knee_joint").id]
    k0 = q_default[idx["knee"][0]]
    a_s = np.linspace(max(knee_lo - k0, -k0) / 2 + 1e-3, min(0.9, (knee_hi - k0) / 2), 80)
    drops = []
    for a in a_s:
        q = _crouch(q_default, idx, a)
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        drops.append(data.site_xpos[foot][2])
    data.qpos[:] = q_default
    mujoco.mj_kinematics(model, data)
    drops = np.array(drops) - data.site_xpos[foot][2]
    order = np.argsort(drops)
    return idx, a_s[order], drops[order]


def _crouch(q, idx, a):
    q = q.copy()
    for k in range(2):
        q[idx["hip_pitch"][k]] -= a
        q[idx["knee"][k]] += 2 * a
        q[idx["ankle_pitch"][k]] -= a
    return q


def _smooth(x, n):
    if n <= 1 or len(x) < 3:
        return x
    k = np.ones(n) / n
    pad = np.pad(x, [(n // 2, n - 1 - n // 2)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, pad)


def _rot_err(R, Rt):
    """Axis-angle vector taking R to Rt (world frame)."""
    E = Rt @ R.T
    a = np.arccos(np.clip((np.trace(E) - 1) / 2, -1, 1))
    if a < 1e-6:
        return np.zeros(3)
    return a / (2 * np.sin(a)) * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])


def _ik(model, data, q, qadr, dofs, lim, body, pt, Rt, w, iters, damping, J):
    """Damped least squares on the joints qadr/dofs so body reaches (pt, Rt). -> (joints, cost)."""
    import mujoco
    qq = q.copy()
    for _ in range(iters):
        data.qpos[:] = qq
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        ep = pt - data.xpos[body]
        er = _rot_err(data.xmat[body].reshape(3, 3), Rt)
        if np.linalg.norm(ep) < 1e-3 and np.linalg.norm(er) < 1e-2:
            break
        mujoco.mj_jacBody(model, data, J[:3], J[3:], body)
        Ja = J[:, dofs] * w[:, None]
        dq = Ja.T @ np.linalg.solve(Ja @ Ja.T + damping ** 2 * np.eye(6), np.r_[ep, er] * w)
        qq[qadr] = np.clip(qq[qadr] + dq, lim[:, 0], lim[:, 1])
    data.qpos[:] = qq
    mujoco.mj_kinematics(model, data)
    cost = np.linalg.norm(pt - data.xpos[body]) + w[3] * np.linalg.norm(_rot_err(data.xmat[body].reshape(3, 3), Rt))
    return qq[qadr], cost


def _fill(x, ok):
    """Linear interpolation over the frames where ok is False (per column)."""
    x = np.array(x, float)
    idx = np.arange(len(x))
    for c in range(x.shape[1]):
        x[:, c] = np.interp(idx, idx[ok], x[ok, c])
    return x


def tracked_feet(take, base_yaw, dt, max_lift=0.3):
    """Foot poses (n, 2, 4): x, y, lift above the planted height, yaw, from PICO's tracked ankles and
    feet (scene frame, G1 scale), or None when the take has no (or too little) leg tracking."""
    body, names = getattr(take, "body", None), getattr(take, "body_names", [])
    idx = {n: k for k, n in enumerate(names)}
    if body is None or any(f"{s}_{j}" not in idx for s in SIDES for j in ("ankle", "foot")):
        return None
    out = np.zeros((take.n, 2, 4))
    w = max(1, int(round(0.05 / dt)))   # ~50 ms: tracker jitter, not steps
    for k, s in enumerate(SIDES):
        A, F = body[:, idx[f"{s}_ankle"]], body[:, idx[f"{s}_foot"]]
        ok = np.all(np.isfinite(A), axis=1) & np.all(np.isfinite(F), axis=1)
        if ok.mean() < 0.8:
            return None
        A, F = _smooth(_fill(A, ok), w), _smooth(_fill(F, ok), w)
        planted = np.percentile(A[:, 2], 10)      # this ankle's height when the foot is down
        d = F[:, :2] - A[:, :2]
        yaw = np.where(np.linalg.norm(d, axis=1) > 0.02, np.arctan2(d[:, 1], d[:, 0]), base_yaw)
        yaw = _smooth(np.unwrap(yaw)[:, None], 3 * w)[:, 0]
        out[:, k] = np.c_[A[:, :2], np.clip(A[:, 2] - planted, 0.0, max_lift), yaw]
    return out


def retarget(model, q_default, take, links, iters=25, rot_weight=0.25, damping=0.03):
    """take: view_take.Take -> dict of per-frame arrays for the viewer."""
    import mujoco
    data = mujoco.MjData(model)
    n = take.n
    dt = float(np.median(np.diff(take.t))) if n > 1 else 1 / 60
    head_site = model.site("head").id
    jid = lambda s, chain: [model.joint(f"{s}_{j}_joint").id for j in chain]
    arm_dofs = {s: model.jnt_dofadr[jid(s, ARM)] for s in SIDES}
    arm_q = {s: model.jnt_qposadr[jid(s, ARM)] for s in SIDES}
    lim = {s: model.jnt_range[jid(s, ARM)] for s in SIDES}
    leg_dofs = {s: model.jnt_dofadr[jid(s, LEG)] for s in SIDES}
    leg_q = {s: model.jnt_qposadr[jid(s, LEG)] for s in SIDES}
    leg_lim = {s: model.jnt_range[jid(s, LEG)] for s in SIDES}
    wrist = {s: model.body(f"{s}_wrist_yaw_link").id for s in SIDES}
    ankle = {s: model.body(f"{s}_ankle_roll_link").id for s in SIDES}
    body_ids = [model.body(name).id for name in links]

    # base: yaw from the smoothed gaze, head site under the (lightly smoothed) tracked head
    yaw = _smooth(np.unwrap(take.yaws), max(1, int(0.8 / dt)))
    head = _smooth(take.heads, max(1, int(0.15 / dt)))
    q = q_default.copy()
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    head_off = data.site_xpos[head_site] - data.qpos[:3]          # pelvis -> head site at yaw 0
    _, _, crouch_drop = _crouch_table(model, data, q_default)       # how far the knees can lower the pelvis
    head_z0 = q_default[2] + head_off[2]
    ankle0 = {s: data.xpos[ankle[s]] - data.qpos[:3] for s in SIDES}  # default ankle in the base frame
    ankle_z0 = data.xpos[ankle["left"]][2]                          # ankle height with the sole on the floor
    # base path, then footsteps planned along it
    Rz = lambda a: np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    base_xy = np.array([head[i, :2] - (Rz(yaw[i]) @ head_off)[:2] for i in range(n)])
    base_z = q_default[2] - np.clip(head_z0 - head[:, 2], crouch_drop[0], crouch_drop[-1])
    feet = tracked_feet(take, yaw, dt)
    if feet is not None:   # the operator's own steps
        swinging, feet_source = feet[:, :, 2] > 0.03, "tracked"
    else:
        feet, swinging = plan_footsteps(take.t, base_xy, yaw, {s: ankle0[s][:2] for s in SIDES})
        feet_source = "planned"
    furniture = np.flatnonzero((model.geom_bodyid == 0) & (model.geom_contype == 2))
    fnames = [model.geom(g).name for g in furniture]
    groups = {f"{s}_{k}": [model.geom(x.format(s=s)).id for x in (v if isinstance(v, tuple) else (v,))]
              for s in SIDES for k, v in GROUPS.items()}
    groups["body"] = [model.geom(x).id for x in BODY_GEOMS]

    out = dict(qpos=np.zeros((n, model.nq)), body_pos=np.zeros((n, len(body_ids), 3)),
               body_quat=np.zeros((n, len(body_ids), 4)))
    out["feet"], out["swing"], out["feet_source"] = feet, swinging, np.array(feet_source)
    for s in SIDES:
        out[f"{s}_foot_err"] = np.zeros(n)
        out[f"{s}_pos_err"] = np.full(n, np.nan)
        out[f"{s}_rot_err"] = np.full(n, np.nan)
        out[f"{s}_target"] = np.full((n, 4, 4), np.nan)
    for g in groups:
        out[f"clear_{g}"] = np.full(n, np.nan)
        out[f"near_{g}"] = np.full(n, -1, int)
    J = np.zeros((6, model.nv))
    fromto = np.zeros(6)
    for i in range(n):
        q[0:2], q[2] = base_xy[i], base_z[i]
        q[3:7] = [np.cos(yaw[i] / 2), 0, 0, np.sin(yaw[i] / 2)]
        leg_w = np.r_[np.ones(3), np.full(3, 0.5)]
        for k, s in enumerate(SIDES):                             # legs: ankle to the planned foot pose
            fx, fy, fz, fyaw = feet[i, k]
            pt = np.r_[fx, fy, ankle_z0 + fz]
            best = _ik(model, data, q, leg_q[s], leg_dofs[s], leg_lim[s], ankle[s], pt, Rz(fyaw), leg_w,
                       iters, damping, J)
            if best[1] > 0.01:
                q2 = q.copy()
                q2[leg_q[s]] = q_default[leg_q[s]]
                cand = _ik(model, data, q2, leg_q[s], leg_dofs[s], leg_lim[s], ankle[s], pt, Rz(fyaw), leg_w,
                           3 * iters, damping, J)
                best = min(best, cand, key=lambda c: c[1])
            q[leg_q[s]] = best[0]
            out[f"{s}_foot_err"][i] = best[1]
        for s in SIDES:
            h = take.hands[s]
            if not h["tracked"][i]:
                continue                                          # hold the last arm pose
            Rt = h["R"][i] @ R_OPENXR_ROBOT @ R_TO_UNITREE_ARM[s]
            pt = h["wrist"][i]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = Rt, pt
            out[f"{s}_target"][i] = T
            def solve(qa, w, n_it):
                qq = q.copy()
                qq[arm_q[s]] = qa
                return _ik(model, data, qq, arm_q[s], arm_dofs[s], lim[s], wrist[s], pt, Rt, w, n_it, damping, J)

            full = np.r_[np.ones(3), np.full(3, rot_weight)]
            best = solve(q[arm_q[s]], full, iters)              # warm start from the previous frame
            if best[1] > 0.03:                                  # stuck (joint limit, elbow flip): reseed
                for seed in (q_default[arm_q[s]], solve(q_default[arm_q[s]], np.r_[np.ones(3), np.full(3, 1e-3)], iters)[0]):
                    cand = solve(seed, full, 3 * iters)
                    if cand[1] < best[1]:
                        best = cand
            q[arm_q[s]] = best[0]
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        for s in SIDES:
            if take.hands[s]["tracked"][i]:
                T = out[f"{s}_target"][i]
                out[f"{s}_pos_err"][i] = np.linalg.norm(T[:3, 3] - data.xpos[wrist[s]])
                out[f"{s}_rot_err"][i] = np.linalg.norm(_rot_err(data.xmat[wrist[s]].reshape(3, 3), T[:3, :3]))
        out["qpos"][i] = q
        out["body_pos"][i] = data.xpos[body_ids]
        out["body_quat"][i] = data.xquat[body_ids]
        # real collision geoms vs furniture (MuJoCo signed distance, capped at 1 m)
        for gname, geoms in groups.items():
            best, who = 1.0, -1
            for g in geoms:
                near = np.linalg.norm(model.geom_aabb[furniture, 3:], axis=1)
                d0 = np.linalg.norm(data.geom_xpos[furniture] - data.geom_xpos[g], axis=1) - near - model.geom_rbound[g]
                for k in np.flatnonzero(d0 < best):
                    d = mujoco.mj_geomDistance(model, data, g, furniture[k], best, fromto)
                    if d < best:
                        best, who = d, k
            out[f"clear_{gname}"][i] = best
            out[f"near_{gname}"][i] = who
    out["yaw"] = yaw
    out["pelvis_drop"] = q_default[2] - out["qpos"][:, 2]
    out["furniture_names"] = np.array(fnames)
    return out
