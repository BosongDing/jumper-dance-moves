"""Retarget the GANimator crab dance onto Jumper -> library/data/ganimator.npz.

The clip is Crab-dance-long.bvh from the GANimator repo (Li et al., SIGGRAPH 2022):
12.8 s of a skeletal crab dancing, 30 fps, with a 0.5 s beat. Their data folder isn't
under their code licence, so we don't redistribute it; this script downloads it from
their repo.

The crab has six walking legs and two claws, Jumper has four legs and two arms, so:
  claws  the direction from each claw's shoulder to its tip, measured in the crab's
         heading frame (so "up" stays up whatever the body does), becomes a palm-pad
         target from Jumper's shoulder, further out the more the crab's claw is
         stretched. The IK reaches for it, one frame after another.
  body   the root's bob, roll, pitch and yaw around their mean, scaled down a lot
         (the crab swings through 70 degrees, Jumper's feet allow about 15).
  legs   not retargeted, Jumper's stay planted.

    python library/ganimator.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from motion.compose import Composer, Station, TempoMap, use_library  # noqa: E402
from motion.robot import ARMS, DANCE, SUPPORT, Costs, Kin, Targets, WholeBodyIK, mirror  # noqa: E402

moves = use_library()
GANIMATOR_BEAT_S, GANIMATOR_NPZ, UP_TALL = moves.GANIMATOR_BEAT_S, moves.GANIMATOR_NPZ, moves.UP_TALL
ganimator_looped = moves.ganimator_looped

URL = "https://raw.githubusercontent.com/PeizhuoLi/ganimator/main/data/Crabnew/Crab-dance-long.bvh"
BVH = ROOT / "library" / "data" / "Crab-dance-long.bvh"
TO_ROBOT = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)  # BVH (left, up, fwd) -> robot (fwd, left, up)
SHOULDER = {"LF": np.array([0.051, 0.061, 0.011]), "RF": np.array([0.051, -0.061, 0.011])}  # base frame
REACH = (0.165, 0.215)        # palm-pad distance from the shoulder: folded (UP) to stretched (OUT)
PLAY_BPM = 125.0
SPEED_CAP = 0.7 * 30.735      # rad/s when played at 125 BPM


def read_bvh(path):
    """Joint names, parents, offsets, channels, frames and frame time of a BVH file."""
    tokens = Path(path).read_text().split()
    names, parents, offsets, channels, stack = [], [], [], [], []
    i = 0
    while tokens[i] != "MOTION":
        t = tokens[i]
        if t in ("ROOT", "JOINT"):
            names.append(tokens[i + 1])
            parents.append(stack[-1] if stack else -1)
            i += 2
        elif t == "End":
            names.append(f"{names[stack[-1]]}_end")
            parents.append(stack[-1])
            channels.append([])
            i += 2
        elif t == "{":
            stack.append(len(names) - 1)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            offsets.append([float(x) for x in tokens[i + 1:i + 4]])
            i += 4
        elif t == "CHANNELS":
            n = int(tokens[i + 1])
            channels.append(tokens[i + 2:i + 2 + n])
            i += 2 + n
        else:
            i += 1
    n_frames, dt = int(tokens[i + 2]), float(tokens[i + 5])
    frames = np.array(tokens[i + 6:], dtype=float).reshape(n_frames, -1)
    return names, np.array(parents), np.array(offsets), channels, frames, dt


def forward_kinematics(parents, offsets, channels, frames):
    """World positions (T, J, 3) and rotations (T, J, 3, 3) of every joint."""
    T, J = len(frames), len(parents)
    local_rot = np.tile(np.eye(3), (T, J, 1, 1))
    local_pos = np.tile(offsets, (T, 1, 1))
    col = 0
    for j, ch in enumerate(channels):
        vals = frames[:, col:col + len(ch)]
        col += len(ch)
        pos = [ch.index(c) for c in ch if c.endswith("position")]
        rot = [c for c in ch if c.endswith("rotation")]
        if pos:
            local_pos[:, j] = vals[:, pos]
        if rot:
            order = "".join(c[0].upper() for c in rot)
            local_rot[:, j] = Rotation.from_euler(order, vals[:, [ch.index(c) for c in rot]],
                                                  degrees=True).as_matrix()
    pos = np.zeros((T, J, 3))
    rot = np.zeros((T, J, 3, 3))
    for j in range(J):
        p = parents[j]
        if p < 0:
            rot[:, j], pos[:, j] = local_rot[:, j], local_pos[:, j]
        else:
            rot[:, j] = rot[:, p] @ local_rot[:, j]
            pos[:, j] = pos[:, p] + np.einsum("tij,tj->ti", rot[:, p], local_pos[:, j])
    return pos, rot


def crab_motion(path):
    """Per frame: each claw's unit direction and stretch (0..1), and the body track
    (height, roll, pitch, yaw) relative to the clip's mean."""
    names, parents, offsets, channels, frames, dt = read_bvh(path)
    pos, rot = forward_kinematics(parents, offsets, channels, frames)
    n = {name: i for i, name in enumerate(names)}
    # Heading: yaw only, from the line between the two claw shoulders (the hip bone's
    # own axes don't point where the crab faces).
    left = pos[:, n["DEF_BN_Arm_L_01"]] - pos[:, n["DEF_BN_Arm_R_01"]]
    left[:, 1] = 0.0
    fwd = np.cross(left, np.array([0.0, 1.0, 0.0]))
    yaw = np.arctan2(fwd[:, 0], fwd[:, 2])
    heading = Rotation.from_euler("y", yaw[:, None]).as_matrix()
    claws = {}
    for arm, side in (("LF", "L"), ("RF", "R")):
        v = pos[:, n[f"DEF_BN_Arm_{side}_03_end"]] - pos[:, n[f"DEF_BN_Arm_{side}_01"]]
        v = np.einsum("tji,tj->ti", heading, v) @ TO_ROBOT.T
        length = np.linalg.norm(v, axis=1)
        claws[arm] = (v / length[:, None], (length - length.min()) / max(np.ptp(length), 1e-9))
    rel = np.einsum("tji,tjk->tik", heading, rot[:, 0])
    ypr = Rotation.from_matrix(np.einsum("ij,tjk,lk->til", TO_ROBOT, rel, TO_ROBOT)).as_euler("ZYX")
    ypr -= ypr.mean(axis=0)
    h = pos[:, 0, 1]
    wobble = np.unwrap(yaw) - np.convolve(np.unwrap(yaw), np.ones(31) / 31, mode="same")   # heading drift removed
    body = np.stack([(h - h.mean()) / max(np.ptp(h), 1e-9), ypr[:, 2], ypr[:, 1], wobble], axis=1)
    return dt, claws, body


def claw_targets(claws, b, dt):
    """Palm-pad targets in Jumper's base frame at beat b."""
    out = {}
    for arm, (d, stretch) in claws.items():
        di = ganimator_looped(d, b, dt)
        di /= np.linalg.norm(di)
        reach = REACH[0] + (REACH[1] - REACH[0]) * float(ganimator_looped(stretch, b, dt))
        out[arm] = SHOULDER[arm] + reach * di
    return out


def keep_reachable(p, arm):
    """In front of the shoulder, above the belly, out of the shell, not across the midline."""
    p = p.copy()
    p[2] = max(p[2], 0.0)
    p[0] = max(p[0], 0.06)
    r = np.hypot(p[0], p[1])
    if r < 0.17 and p[2] < 0.10:
        p[:2] *= 0.17 / max(r, 1e-6)
    side = 1.0 if arm == "LF" else -1.0
    if side * p[1] < -0.02:
        p[1] = -0.02 * side
    return p


def main():
    if not BVH.is_file():
        print(f"downloading {URL}")
        BVH.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, BVH)
    dt, claws, body = crab_motion(BVH)

    stance = Composer([Station(2, "still", "hold_up", k_override=0.0, blend=0.0)],
                      TempoMap.constant(125, 8), limits={}).run().qpos[-1].copy()
    ik = WholeBodyIK(DANCE.leg_q, costs=Costs(claw_tip=15.0, arm_posture_soft=0.3))
    ik.reset(stance)
    kin = Kin()
    kin.set(stance)
    feet = {s: kin.site_pos(s) for s in SUPPORT}
    R, P = kin.data.xmat[kin.body["base_link"]].reshape(3, 3), kin.body_pos("base_link")
    arm_q = np.zeros((len(body), 2, 5))
    err = np.zeros((len(body), 2))
    for i in range(len(body)):
        tips = claw_targets(claws, i * dt / GANIMATOR_BEAT_S, dt)
        tg = Targets(base_pos=stance[:3], base_quat=stance[3:7], feet=feet,
                     arm_q={"LF": UP_TALL.copy(), "RF": mirror(UP_TALL)},
                     claw_tip={a: P + R @ keep_reachable(tips[a], a) for a in ARMS})
        q, e = ik.solve(tg, iters=60)
        arm_q[i] = q[7:12], q[12:17]
        err[i] = e["tip"]["LF"], e["tip"]["RF"]

    # smooth, then keep smoothing until it's under the servo speed at 125 BPM
    play_dt = dt / (GANIMATOR_BEAT_S / (60.0 / PLAY_BPM))
    arm_q = gaussian_filter1d(arm_q, sigma=1.5, axis=0, mode="wrap")
    for _ in range(50):
        if np.abs(np.diff(arm_q, axis=0)).max() / play_dt <= SPEED_CAP:
            break
        arm_q = gaussian_filter1d(arm_q, sigma=1.0, axis=0, mode="wrap")
    arm_q[:, :, 4] = np.clip(arm_q[:, :, 4], [-1.5, -0.05], [0.05, 1.5])
    np.savez_compressed(GANIMATOR_NPZ, dt=dt, arm_q=arm_q, body=body)
    print(f"{len(body)} frames, claw tips within {np.median(err) * 1000:.1f} mm (median), "
          f"{np.percentile(err, 95) * 1000:.1f} mm (95%) -> {GANIMATOR_NPZ}")


if __name__ == "__main__":
    main()
