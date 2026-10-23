"""Solve the four keyframes of the dig (the rake in the sand, MV 0:37).

Most claw keyframes only have to look right, so they were picked by eye. The dig has
a job: the palm pad has to hit the sand in front, drag back along it, lift and wind
up again, all while the body is crouched. So those four poses are solved, and the
result is pasted into DIG_KEYS in layers.py (the pincer column is set by hand after).

    python library/solve_dig.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from compose import Composer, Station, TempoMap, use_library  # noqa: E402
from robot import FOOT_SITE_Z, MIRROR, SUPPORT, Kin, polygon_margin  # noqa: E402

moves = use_library()
DRUM_HIT, UP = moves.DRUM_HIT, moves.UP

GROUND = FOOT_SITE_Z + 0.002


def crouched_pose():
    c = Composer([Station(2, "crouch", "hold_up", k_override=1.0, blend=0.0)],
                 TempoMap.constant(125, 8), limits={})
    return c.run().qpos[-1].copy()


def solve_pad(k, q0, target, seed, reg=0.02):
    """Left arm joints J0..J3 that put the palm pad on `target`, clear of the shell."""
    def pose(a):
        q = q0.copy()
        q[7:11] = a
        q[12:17] = UP * MIRROR
        k.set(q)

    def cost(a):
        pose(a)
        miss = k.site_pos("LF") - target
        too_close = max(0.0, 0.006 - k.chassis_clearance("LF"))
        return 1e4 * float(miss @ miss) + 1e5 * too_close ** 2 + reg * float(np.sum((a - seed[:4]) ** 2))

    r = minimize(cost, seed[:4], method="Nelder-Mead", options=dict(maxiter=2000, xatol=1e-4, fatol=1e-8))
    pose(r.x)
    margin = polygon_margin(np.array([k.site_pos(s)[:2] for s in SUPPORT]), k.com()[:2])
    print(f"    off by {np.linalg.norm(k.site_pos('LF') - target) * 1000:.1f} mm, "
          f"{k.chassis_clearance('LF') * 1000:.1f} mm from the shell, COM margin {margin * 1000:.0f} mm")
    return r.x


def main():
    k = Kin()
    q0 = crouched_pose()
    base = q0[:3] * [1, 1, 0]
    targets = {                       # left palm pad, relative to the body, metres
        "hit": [0.215, 0.090, GROUND],
        "rake": [0.140, 0.115, GROUND],
        "lift": [0.150, 0.130, 0.060],
        "up": [0.215, 0.110, 0.115],
    }
    seed = DRUM_HIT.copy()
    rows = []
    for name, t in targets.items():
        print(name)
        a = solve_pad(k, q0, base + np.array(t), seed)
        rows.append(a)
        seed = np.append(a, 0.0)
    print("DIG_KEYS = np.array([" + ",\n                     ".join(
        "[" + ", ".join(f"{x:.3f}" for x in a) + ", 0.0]" for a in rows) + "])")


if __name__ == "__main__":
    main()
