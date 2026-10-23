"""The layers every move in the library is made of.

A move (library/moves.yaml) is a body layer plus a claw layer, sometimes with an accent:

  body    how the shell moves over the four planted legs (BODY)
  claws   what the two arms do (CLAWS)
  accent  small extras on the beat: pincer snaps, a dip (ACCENTS)

Each layer is a function of the beat count, made from a description of a moment of the
Crab Rave music video ("MV 1:26" says where to look) and tuned by watching the result.
To add your own, see SKILL.md.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from motion.compose import ClawOut
from motion.curves import bumps, catmull_rom_loop, frac, hits, pulse, smoothstep, strike, swing, swing_sin
from motion.robot import arm_tips_base, base_pose, mirror, quat_mat

DATA = Path(__file__).resolve().parent / "data"

# The dig's keyframes were solved for a full crouch, so it's never danced less crouched.
NEEDS = {"dig": ("crouch", 1.0), "dig_fast": ("crouch", 1.0)}
# Danced at the size written below, not measured (only used together, in the video).
FIXED = ("duet_slow", "duet_fast")
# Made from someone else's data: not in the shared reference files.
NOT_SHARED = ("ganimator",)


# ---------------------------------------------------------------------------
# Body moves
#
# Each returns the base offset [dx, dy, dz, roll, pitch, yaw] from the DANCE stance
# (m, rad; roll + is left side up, pitch + is nose down). The numbers in BODY are the
# shape at k = 1; limits.yaml says how far each one can be scaled on the real robot,
# and the amplitude actually danced is intensity * k_max * shape.
# ---------------------------------------------------------------------------

def _still(b, a, period):
    return np.zeros(6)


def _bounce(b, a, period):
    # dip on every beat, lowest right on it, with a little nod
    d = (0.5 + 0.5 * swing(b, 1.0)) ** 1.5
    return np.array([0.0, 0.0, -a["z"] * d, 0.0, a["pitch"] * d, 0.0])


def _swing_x(b, a, period):
    # forward on even beats, back on odd ones, leaning into it
    c = swing(b, period)
    return np.array([a["x"] * c, 0.0, -a["z"] * abs(c), 0.0, a["pitch"] * c, 0.0])


def _swing_y(b, a, period):
    # side to side, rolling into the side it moves to
    c = swing(b, period)
    return np.array([0.0, a["y"] * c, -a["z"] * abs(c), -a["roll"] * c, 0.0, 0.0])


def _circle(b, a, period):
    # forward -> left -> back -> right, tilting with it
    c, s = swing(b, period), swing_sin(b, period)
    return np.array([a["x"] * c, a["y"] * s, 0.0, -a["roll"] * s, a["pitch"] * c, 0.0])


def _crouch(b, a, period):
    # the digging posture (MV 0:37): low, nose down, a bob on every stroke and a lean
    # toward the claw that's striking
    bob = 0.5 + 0.5 * swing(b, period / 2.0)
    lean = swing(b, period)
    return np.array([0.0, 0.0, -a["z"] - a["bob"] * bob, -a["roll"] * lean, a["pitch"], 0.0])


def _arch(b, a, period):
    # MV 1:26: forward and down, up through the middle, back and down
    c = swing(b, period)
    return np.array([a["x"] * c, 0.0, -a["z"] * abs(c) ** 1.5, 0.0, a["pitch"] * c, 0.0])


def _arch_y(b, a, period):
    # the same arch, side to side
    c = swing(b, period)
    return np.array([0.0, a["y"] * c, -a["z"] * abs(c) ** 1.5, -a["roll"] * c, 0.0, 0.0])


def _hcircle_big(b, a, period):
    # level circle, no tilt (MV 1:26-1:29). An oval: the feet are 0.36 m apart side to
    # side but only 0.12 m front to back, and it's centred a bit back so the front of
    # the circle doesn't run the COM off the front feet.
    c, s = swing(b, period), swing_sin(b, period)
    return np.array([-a["back"] + a["x"] * c, a["y"] * s, 0.0, 0.0, 0.0, 0.0])


def _low_sweep(b, a, period):
    # sink on the left, travel low to the right, rise there; then the mirror image
    n = int(np.floor(b / period))
    u = b / period - n
    side = 1.0 if n % 2 == 0 else -1.0
    down = smoothstep(0.0, 0.25, u) * (1.0 - smoothstep(0.75, 1.0, u))
    cross = smoothstep(0.2, 0.8, u)
    y = side * a["y"] * (1.0 - 2.0 * cross)
    return np.array([0.0, y, -a["z"] * down, -a["roll"] * y / max(a["y"], 1e-6), 0.0, 0.0])


def _low_glide(b, a, period):
    # the low sweep at double speed, re-timed so it leaves and arrives on beats and is
    # lowest on the beat in between
    n = int(np.floor(b / period))
    u = b / period - n
    side = 1.0 if n % 2 == 0 else -1.0
    down = smoothstep(0.0, 0.3, u) * (1.0 - smoothstep(0.7, 1.0, u))
    y = side * a["y"] * (1.0 - 2.0 * smoothstep(0.0, 1.0, u))
    return np.array([0.0, y, -a["z"] * down, -a["roll"] * y / max(a["y"], 1e-6), 0.0, 0.0])


def _lean_nod(b, a, period):
    # MV 1:13: reared up, nodding left on one beat and right on the next
    side = swing(b, period)
    return np.array([-a["x"], 0.0, a["z"], -a["roll"] * side, -a["pitch"], a["yaw"] * side])


def _lean_shake(b, a, period):
    # reared up, pumping down hard on every beat
    d = (0.5 + 0.5 * swing(b, period)) ** 1.5
    return np.array([-a["x"], 0.0, a["z"] - a["bob"] * d, 0.0, -a["pitch"], 0.0])


def _rise(b, a, period):
    # MV 1:07: crouch, then stand up tall with the head raised, top on the beat.
    # The feet never leave the ground. Pairs with rise_clap.
    up = (0.5 + 0.5 * swing(b, period)) ** 1.3
    return np.array([-a["x"] * up, 0.0, -a["crouch"] + (a["crouch"] + a["tall"]) * up, 0.0,
                     -a["pitch"] * up, 0.0])


def _rise_hits(b, a, period, positions):
    # rise on an irregular rhythm: up at each hit, crouched and bobbing in between so
    # it never stands still
    up = hits(b, positions, period, before=0.8, after=0.9, sharp=None)
    d = (0.5 + 0.5 * swing(b, 1.0)) ** 1.5
    wait = -a["crouch"] + 0.3 * a["crouch"] * (1.0 - d)
    return np.array([-a["x"] * up, 0.0, wait * (1.0 - up) + a["tall"] * up, 0.0,
                     -a["pitch"] * up + 0.25 * a["pitch"] * d * (1.0 - up), 0.0])


# The opening clap duet. The song claps on 1, 2, 3, 3-and, 4: the slow crab takes the
# two big claps on 1 and 2, the fast one answers with three quick ones. Each stays low
# while the other claps, bobbing and turning its head toward its partner.
SLOW_HITS, FAST_HITS = (0.0, 1.0), (2.0, 2.5, 3.0)


def _duet(b, a, period, positions, before, after, wait_bob):
    up = hits(b, positions, period, before=before, after=after, sharp=None)
    d = (0.5 + 0.5 * swing(b, 1.0)) ** 1.5
    wait = -a["crouch"] + wait_bob * a["crouch"] * (1.0 - d)
    return np.array([-a["x"] * up, 0.0, wait * (1.0 - up) + a["tall"] * up, 0.0,
                     -a["pitch"] * up + 0.2 * a["pitch"] * d * (1.0 - up), a["look"] * (1.0 - up)])


def _duet_slow(b, a, period):
    return _duet(b, a, period, SLOW_HITS, 1.0, 1.2, 0.25)


def _duet_fast(b, a, period):
    return _duet(b, a, period, FAST_HITS, 0.6, 0.8, 0.3)


def _ganimator_body(b, a, period):
    return ganimator_body(b / period, a["k"])


BODY = {
    # name: (function, shape at k = 1, default period in beats)
    "still": (_still, {}, 1.0),
    "bounce": (_bounce, {"z": 0.035, "pitch": 0.12}, 1.0),
    "swing_x": (_swing_x, {"x": 0.032, "z": 0.008, "pitch": 0.22}, 2.0),
    "swing_y": (_swing_y, {"y": 0.034, "z": 0.008, "roll": 0.27}, 2.0),
    "circle": (_circle, {"x": 0.032, "y": 0.034, "roll": 0.24, "pitch": 0.2}, 4.0),
    "crouch": (_crouch, {"z": 0.022, "bob": 0.006, "roll": 0.08, "pitch": 0.12}, 1.0),
    "arch": (_arch, {"x": 0.03, "z": 0.03, "pitch": 0.15}, 2.0),
    "arch_y": (_arch_y, {"y": 0.032, "z": 0.03, "roll": 0.2}, 2.0),
    "hcircle_big": (_hcircle_big, {"back": 0.02, "x": 0.03, "y": 0.05}, 4.0),
    "low_sweep": (_low_sweep, {"y": 0.04, "z": 0.035, "roll": 0.06}, 4.0),
    "low_glide": (_low_glide, {"y": 0.04, "z": 0.035, "roll": 0.06}, 2.0),
    "lean_nod": (_lean_nod, {"x": 0.015, "z": 0.02, "roll": 0.26, "pitch": 0.35, "yaw": 0.12}, 2.0),
    "lean_shake": (_lean_shake, {"x": 0.02, "z": 0.03, "bob": 0.03, "pitch": 0.4}, 1.0),
    "rise": (_rise, {"crouch": 0.03, "tall": 0.025, "pitch": 0.3, "x": 0.01}, 2.0),
    "ganimator": (_ganimator_body, {"k": 1.0}, 1.0),
    # these two aren't in limits.yaml: the shape is the amplitude, dance them at 1
    "duet_slow": (_duet_slow, {"crouch": 0.042, "tall": 0.035, "pitch": 0.42, "x": 0.014, "look": 0.15}, 4.0),
    "duet_fast": (_duet_fast, {"crouch": 0.036, "tall": 0.024, "pitch": 0.28, "x": 0.008, "look": -0.15}, 4.0),
}
BODY_HITS = {"rise": _rise_hits}


# ---------------------------------------------------------------------------
# Claw moves
#
# Keyframes are joint angles J0..J4 of the LEFT arm; the right arm is its mirror
# image (robot.mirror). The look-only ones (UP, V, OUT...) were picked by rendering joint sweeps; the
# ones that touch something (DRUM_HIT, CLAP_CLOSED, DIG_*) were solved, see
# solve_dig.py.
# ---------------------------------------------------------------------------

UP = np.array([-1.22, -3.14, -0.96, -0.35, -0.26])           # straight up (MV 1:13)
UP_HIGH = np.array([-1.0, -3.3, -0.4, -0.2, -0.6])           # up and out, pincers open
UP_TALL = np.array([-1.1, -3.25, -0.75, -0.15, -0.5])        # a little higher than UP
PUMP_LOW = np.array([-0.8, -3.3, -1.4, -0.3, -0.1])          # pulled down beside the head
V = np.array([-0.6, -3.8, -0.5, -0.2, -0.8])                 # wide V
OUT = np.array([-0.6, -4.5, -0.2, 0.0, -0.4])                # straight out sideways (MV 1:34)
CLAP_CLOSED = np.array([-1.826, -3.009, -1.238, -0.405, 0.0])  # tips together over the head
DRUM_UP = np.array([-0.76, -2.05, -0.45, -0.55, -0.4])
DRUM_HIT = np.array([-0.757, -0.622, -0.703, -0.697, 0.0])   # pad on the sand 0.2 m ahead
FORWARD = np.array([-0.55, -1.75, -0.5, -0.7, -0.4])         # both straight ahead
FORWARD_UP = np.array([-0.6, -2.0, -0.6, -0.3, -0.4])        # ahead, angled up
COUNTER_LOW = np.array([-0.8, -2.75, -0.55, -0.45, -0.3])
COUNTER_HIGH = np.array([-1.0, -3.35, -0.55, -0.2, -0.6])

# The rake (MV 0:37), solved crouched: hit the sand 0.215 m ahead, drag back to
# 0.14 m, lift, wind up. The pincers (last column) open a little on the way up, by hand.
DIG_KEYS = np.array([[-0.512, -1.193, -0.642, -0.853, 0.0],
                     [0.312, -1.133, -0.623, -1.638, 0.0],
                     [0.395, -1.568, -0.606, -1.683, -0.2],
                     [-0.384, -1.960, -0.546, -1.033, -0.35]])
DIG_TIMES = np.array([0.0, 0.30, 0.55, 0.80])   # within one stroke
DIG_GROUND = (0.0, 0.30)                         # on the sand from the hit to the end of the drag


def _lerp(a, b, t):
    return a + (b - a) * t


def _hold(pose):
    def f(b, s, period):
        return ClawOut(pose.copy(), mirror(pose))
    return f


def _pump(b, s, period):
    # raise the roof, pushing up on the beat (MV 1:13)
    low = _lerp(UP, PUMP_LOW, s)
    h = strike(b, period, recover=0.45, windup=0.55, sharp=2.0)
    q = _lerp(low, _lerp(UP, UP_HIGH, s), h)
    return ClawOut(q, mirror(q))


def _alternate(b, s, period):
    # one claw high while the other is out, swapping each beat (MV 1:34)
    a = 0.5 + 0.5 * swing(b, period)
    out = _lerp(UP, OUT, s)
    return ClawOut(_lerp(out, UP_TALL, a), mirror(_lerp(out, UP_TALL, 1.0 - a)))


def _clap(b, s, period):
    # open wide off the beat, tips together over the head on it
    h = strike(b, period, recover=0.4, windup=0.5, sharp=2.5)
    q = _lerp(_lerp(UP, V, s), CLAP_CLOSED, h)
    return ClawOut(q, mirror(q))


def _contact(h):
    return float(np.clip((h - 0.8) / 0.2, 0.0, 1.0))


def _drum(b, s, period):
    # alternate hits on the sand, left on even beats, right on odd (MV 0:36)
    p = 2.0 * period
    up = _lerp(DRUM_HIT, DRUM_UP, 0.5 + 0.5 * s)
    hl = strike(b, p, 0.0, recover=0.3, windup=0.45, sharp=2.5)
    hr = strike(b, p, p / 2.0, recover=0.3, windup=0.45, sharp=2.5)
    return ClawOut(_lerp(up, DRUM_HIT, hl), mirror(_lerp(up, DRUM_HIT, hr)),
                   {"LF": _contact(hl), "RF": _contact(hr)})


def _dig(b, s, period):
    # the rake: each claw strikes once per period, the right half a cycle behind, so
    # the sand gets hit every half period
    out, ground = [], {}
    mean = DIG_KEYS.mean(axis=0)
    for arm, off in (("LF", 0.0), ("RF", 0.5)):
        u = frac(b / period - off)
        q = catmull_rom_loop(DIG_KEYS, DIG_TIMES, u)
        q = mean + (q - mean) * (0.6 + 0.4 * s)
        g0, g1 = DIG_GROUND
        ground[arm] = 1.0 if g0 <= u <= g1 else max(0.0, 1.0 - smoothstep(g1, g1 + 0.06, u)) \
            if u > g1 else smoothstep(0.94, 1.0, u)
        out.append(q)
    return ClawOut(out[0], mirror(out[1]), ground)


def _counter(b, s, period):
    # against the body: up when it dips, down when it rises (MV 2:33)
    d = (0.5 + 0.5 * swing(b, period)) ** 1.5
    q = _lerp(COUNTER_LOW, _lerp(COUNTER_LOW, COUNTER_HIGH, s), d)
    return ClawOut(q, mirror(q))


def _groove(b, s, period):
    # claws up, keeping time with a small bob, the right one a quarter beat behind
    out = []
    for off in (0.0, 0.25):
        d = 0.5 + 0.5 * swing(b - off, period)
        q = UP.copy()
        q[1] += -0.30 * s * (d - 0.5)
        q[2] += 0.35 * s * (d - 0.5)
        q[4] = -0.2 - 0.3 * s * d
        out.append(q)
    return ClawOut(out[0], mirror(out[1]))


def _swing_fwd(b, s, period):
    # with arch / swing_x: up and back when the body is back, forward and down when
    # it's forward
    c = 0.5 + 0.5 * swing(b, period)
    q = _lerp(UP_TALL, _lerp(UP_TALL, FORWARD, 0.75 * s), c)
    q[4] = -0.2 - 0.4 * c
    return ClawOut(q, mirror(q))


def _swing_side(b, s, period):
    # with arch_y / swing_y: both claws lean toward the side the body moves to
    c = swing(b, period)
    ql, qr = UP.copy(), UP.copy()
    ql[0] += -0.45 * s * c
    qr[0] += 0.45 * s * c
    ql[1] += 0.15 * s * abs(c)
    qr[1] += 0.15 * s * abs(c)
    return ClawOut(ql, mirror(qr))


def _up_slap(b, s, period):
    # both claws from straight ahead to straight up, slapping up on the beat
    h = strike(b, period, recover=0.45, windup=0.5, sharp=2.5)
    q = _lerp(_lerp(UP_TALL, FORWARD, s), UP_TALL, h)
    q[4] = -0.6 * h
    return ClawOut(q, mirror(q))


def _low_open(s):
    # claws down beside the body, pincers open
    q = _lerp(UP, OUT, 0.6 * s)
    q[0] = _lerp(UP, V, 0.8)[0]
    return q


def _rise_clap(b, s, period):
    # with rise: low and open in the crouch, swinging up, closed over the head at the
    # top (on the beat), open again on the way down
    h = strike(b, period, recover=0.2, windup=0.65, sharp=2.0)
    q = _lerp(_low_open(s), CLAP_CLOSED, h)
    q[4] = -0.8 * (1.0 - h)
    return ClawOut(q, mirror(q))


def _rise_clap_hits(b, s, period, positions):
    # rise_clap on an irregular rhythm: closed over the head at each hit, low and
    # snapping on every beat in between
    h = hits(b, positions, period, before=0.6, after=0.4, sharp=2.0, hold=0.12)
    low = _low_open(s)
    d = 0.5 + 0.5 * swing(b, 1.0)
    low[1] += -0.25 * s * (d - 0.5)
    low[2] += 0.3 * s * (d - 0.5)
    q = _lerp(low, CLAP_CLOSED, h)
    q[4] = -0.8 * (1.0 - h) * (1.0 - 0.6 * pulse(b, 1.0, width=0.4))
    return ClawOut(q, mirror(q))


def _duet_slow_claws(b, s, period):
    # two big claps on 1 and 2, then arms open and down while the partner claps, the
    # pincers snapping along with its three
    u = b % period
    h = hits(b, SLOW_HITS, period, before=0.6, after=0.35, sharp=2.0, hold=0.1)
    low = smoothstep(1.3, 2.2, u) * (1.0 - smoothstep(3.0, 3.6, u))
    q = _lerp(_lerp(_lerp(UP, V, 0.7 * s), _low_open(s), low), CLAP_CLOSED, h)
    q[4] = -0.8 * (1.0 - h) * (1.0 - 0.7 * low * bumps(b, FAST_HITS, period, 0.12))
    return ClawOut(q, mirror(q))


def _duet_fast_claws(b, s, period):
    # low and coiled during the partner's claps, arms up just before 3, three quick
    # small claps, down again
    u = b % period
    h = hits(b, FAST_HITS, period, before=0.22, after=0.2, sharp=1.5, hold=0.04)
    ready = smoothstep(1.2, 1.8, u) * (1.0 - smoothstep(3.3, 4.0, u))
    near = _lerp(CLAP_CLOSED, UP_HIGH, 0.45 * s)
    q = _lerp(_lerp(_low_open(s), near, ready), CLAP_CLOSED, h)
    q[4] = -0.8 * (1.0 - h) * (1.0 - 0.7 * (1.0 - ready) * bumps(b, SLOW_HITS, period, 0.12))
    return ClawOut(q, mirror(q))


def forward_tips_world(arm_left, shift_x=0.0):
    """Where the palm pads are, in the world, for this claw pose with the body at the
    stance centre (moved back by shift_x)."""
    pos, quat = base_pose(np.array([shift_x, 0.0, 0.0, 0.0, 0.0, 0.0]))
    R = quat_mat(quat)
    tips = arm_tips_base({"LF": arm_left, "RF": mirror(arm_left)})
    return {a: pos + R @ p for a, p in tips.items()}


def _isolation_big(b, s, period):
    # MV 1:41: claws held out in front, (almost) fixed in space while the body circles
    # under them. They follow 35% of the body so the big circle stays within reach.
    return ClawOut(FORWARD_UP.copy(), mirror(FORWARD_UP),
                   tip_world=forward_tips_world(FORWARD_UP, shift_x=-0.02),
                   tip_follow=0.35, tip_center=(-0.02, 0.0))


def _ganimator_claws(b, s, period):
    left, right = ganimator_arms(b / period, s)
    return ClawOut(left, right)


CLAWS = {
    # name: (function(b, intensity, period), default period in beats)
    "hold_up": (_hold(UP), 1.0),
    "hold_v": (_hold(V), 1.0),
    "pump": (_pump, 1.0),
    "alternate": (_alternate, 2.0),
    "clap": (_clap, 1.0),
    "drum": (_drum, 1.0),
    "dig": (_dig, 1.0),           # 8th notes
    "dig_fast": (_dig, 0.5),      # 16th notes
    "counter": (_counter, 1.0),
    "groove": (_groove, 1.0),
    "up_slap": (_up_slap, 1.0),
    "swing_fwd": (_swing_fwd, 2.0),
    "swing_side": (_swing_side, 2.0),
    "rise_clap": (_rise_clap, 2.0),
    "isolation_big": (_isolation_big, 1.0),
    "ganimator": (_ganimator_claws, 1.0),
    "duet_slow": (_duet_slow_claws, 4.0),
    "duet_fast": (_duet_fast_claws, 4.0),
}
CLAW_HITS = {"rise_clap": _rise_clap_hits}


# ---------------------------------------------------------------------------
# GANimator crab dance
#
# A 12.8 s skeletal crab dance from the GANimator paper, retargeted onto Jumper by
# library/ganimator.py (it downloads the .bvh, which isn't ours to share). Its
# beat is 0.5 s, and we play it locked to the music's beat: one of its beats per beat.
# ---------------------------------------------------------------------------

GANIMATOR_NPZ = DATA / "ganimator.npz"
GANIMATOR_BEAT_S = 0.5
GANIMATOR_LOOP_BEATS = 24       # 12 s of the 12.8 s clip, looped with a short crossfade
GANIMATOR_FADE_S = 0.4
GANIMATOR_BODY_GAIN = {"z": 0.06, "roll": 0.25, "pitch": 0.25, "yaw": 0.25}


@lru_cache(maxsize=1)
def _ganimator():
    if not GANIMATOR_NPZ.is_file():
        raise FileNotFoundError(f"{GANIMATOR_NPZ} is missing; run python library/ganimator.py first")
    z = np.load(GANIMATOR_NPZ)
    return float(z["dt"]), z["arm_q"], z["body"]


def _sample(arr, t, dt):
    x = t / dt
    i = int(np.floor(x))
    w = x - i
    return arr[np.clip(i, 0, len(arr) - 1)] * (1 - w) + arr[np.clip(i + 1, 0, len(arr) - 1)] * w


def ganimator_looped(arr, b, dt):
    loop = GANIMATOR_LOOP_BEATS * GANIMATOR_BEAT_S
    t = (b * GANIMATOR_BEAT_S) % loop
    x = _sample(arr, t, dt)
    if t < GANIMATOR_FADE_S:
        w = t / GANIMATOR_FADE_S
        x = _sample(arr, t + loop, dt) * (1 - w) + x * w
    return x


def ganimator_arms(b, intensity=1.0):
    dt, track, _ = _ganimator()
    q = ganimator_looped(track.reshape(len(track), -1), b, dt).reshape(2, 5)
    rest = np.stack([UP_TALL, mirror(UP_TALL)])
    q = rest + (q - rest) * intensity
    return q[0], q[1]


def ganimator_body(b, scale):
    dt, _, body = _ganimator()
    z, roll, pitch, yaw = ganimator_looped(body, b, dt)
    g = GANIMATOR_BODY_GAIN
    return np.array([0.0, 0.0, g["z"] * z * scale, g["roll"] * roll * scale,
                     g["pitch"] * pitch * scale, g["yaw"] * yaw * scale])


# ---------------------------------------------------------------------------
# Accents: each takes the body offset and the claws and returns the new body offset
# ---------------------------------------------------------------------------

GRIP_OPEN = 0.9   # rad


def _snap(b, s, off, claws):
    # pincers shut on the beat and open in between (MV 1:26)
    h = strike(b, 1.0, recover=0.3, windup=0.55, sharp=3.0)
    g = GRIP_OPEN * s * (1.0 - h)
    claws.left[4] = -g
    claws.right[4] = g
    return off


def _dip(b, s, off, claws):
    d = (0.5 + 0.5 * swing(b, 1.0)) ** 1.5
    return off + np.array([0.0, 0.0, -0.025 * s * d, 0.0, 0.06 * s * d, 0.0])


ACCENTS = {"snap": _snap, "bounce": _dip}


# Moves whose data isn't here yet.
UNAVAILABLE = () if GANIMATOR_NPZ.is_file() else ("ganimator",)
