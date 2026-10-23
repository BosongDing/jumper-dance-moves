"""Beat-shaped curves: the building blocks every move is made of.

`b` is the beat count as a float (beat 3 starts at b = 3.0). Each curve says where
something should be at beat b, so a move built from them follows any tempo.

  swing    smooth back and forth, extremes on the beats (sways, circles)
  pulse    a short bump on the beat
  strike   a hit on the beat, a quick release, an accelerating wind-up (claps, drums)
  hits     strike for an irregular rhythm, e.g. claps on 1, 2, 3, 3-and, 4
  bumps    small bumps on chosen beats of the bar (pincer snaps)
  catmull_rom_loop   a smooth loop through a few keyframe poses (the dig)
"""

from __future__ import annotations

import numpy as np


def frac(x):
    return x - np.floor(x)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


def smootherstep(x):
    t = float(np.clip(x, 0.0, 1.0))
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def swing(b, period=2.0, offset=0.0):
    """+1 at b = offset, -1 half a period later."""
    return float(np.cos(2.0 * np.pi * (b - offset) / period))


def swing_sin(b, period=2.0, offset=0.0):
    return float(np.sin(2.0 * np.pi * (b - offset) / period))


def strike(b, period=1.0, offset=0.0, recover=0.35, windup=0.5, sharp=2.5):
    """A drum stroke or a clap: 1 on the hit, back off quickly, then an accelerating
    wind-up into the next hit."""
    u = frac((b - offset) / period)
    if u < windup:
        return 1.0 - smoothstep(0.0, recover, u)
    return float(((u - windup) / (1.0 - windup)) ** sharp)


def pulse(b, period=1.0, offset=0.0, width=0.5):
    """A smooth bump on the beat, zero for the rest of the cycle."""
    u = frac((b - offset) / period + 0.5) - 0.5
    x = abs(u) / (width / 2.0)
    return float(0.5 * (1.0 + np.cos(np.pi * x))) if x < 1.0 else 0.0


def hits(b, positions, cycle=4.0, before=0.5, after=0.4, sharp=2.0, hold=0.0):
    """`strike` for an irregular rhythm, e.g. claps on (0, 1, 2, 2.5, 3) of each bar.

    Ramps up over `before` beats into each hit, falls over `after` beats, and stays at 1
    for `hold` beats after it (so a policy that lags a little still gets all the way
    there). sharp=None gives a smoothstep approach that arrives at rest, for the body.
    """
    v = 0.0
    u = b % cycle
    for p in positions:
        for d in (u - p - cycle, u - p, u - p + cycle):
            if -before <= d <= 0.0:
                x = (d + before) / before
                v = max(v, smoothstep(0.0, 1.0, x) if sharp is None else x ** sharp)
            elif 0.0 < d <= hold:
                v = 1.0
            elif hold < d <= hold + after:
                v = max(v, 1.0 - smoothstep(hold, hold + after, d))
    return float(v)


def bumps(b, positions, cycle=4.0, half=0.12):
    """Short bumps at the given beats of every bar, for small things like pincer snaps."""
    u = b % cycle
    v = 0.0
    for p in positions:
        for d in (u - p - cycle, u - p, u - p + cycle):
            if abs(d) < half:
                v = max(v, 0.5 * (1.0 + np.cos(np.pi * d / half)))
    return float(v)


def split_hits(name):
    """'rise@0,1' -> ('rise', (0.0, 1.0)): a move re-timed to hit on those beats of each
    bar. Plain names give (name, None)."""
    if "@" not in name:
        return name, None
    base, spec = name.split("@", 1)
    return base, tuple(float(x) for x in spec.split(",") if x.strip())


def catmull_rom_loop(points, times, u):
    """Closed Catmull-Rom spline through keyframes `points` (K, D) placed at `times`
    (K,) in [0, 1); u is the position in the cycle. Hits every keyframe exactly, with
    smooth velocity through it. Used for the dig, which is four poses in a loop."""
    P = np.asarray(points, dtype=float)
    T = np.asarray(times, dtype=float)
    K = len(P)
    u = frac(u)
    k = int(np.searchsorted(T, u, side="right") - 1) % K
    t0 = T[k]
    t1 = T[(k + 1) % K] + (1.0 if k + 1 >= K else 0.0)
    if u < t0:
        u += 1.0

    def at(i):
        return P[i % K]

    def tm(i):
        return T[i % K] + np.floor(i / K)

    def tangent(i):
        return (at(i + 1) - at(i - 1)) / (tm(i + 1) - tm(i - 1))

    d = t1 - t0
    s = (u - t0) / d
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    return h00 * at(k) + h10 * d * tangent(k) + h01 * at(k + 1) + h11 * d * tangent(k + 1)
