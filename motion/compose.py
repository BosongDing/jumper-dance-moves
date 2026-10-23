"""From a list of moves on the beat to a joint trajectory the policy can track.

A timeline is a list of segments laid end to end, measured in beats:

  HoldHome  stand in the walking stance
  Enter     step the four legs in under the body, then raise the claws
  Station   one move: body + claws + accents
  Exit      lower the claws to the sand, step the legs back out

A TempoMap turns beats into seconds, so the same timeline plays at any tempo, even one
that drifts. Each frame's pose is solved by the whole-body IK in robot.py.

The moves themselves come from the library: library/layers.py (how each body and claw
layer moves), library/moves.yaml (the named moves) and library/limits.yaml (how big each
may get).
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import yaml

from motion.curves import smootherstep, split_hits
from motion.robot import ARMS, DANCE, HOME_STANCE, JOINTS, SUPPORT, Kin, Targets, WholeBodyIK, arm_tips_base, \
    base_pose, home_qpos, quat_from_rpy, quat_mat, rpy_from_quat

ROOT = Path(__file__).resolve().parents[1]
CONTROL_DT = 0.02          # 50 Hz, the dance policy's rate
STEP_LIFT = 0.03           # m, foot height while stepping in / out
STEP_ORDER = ("LR", "RR", "LM", "RM")


# ---------------------------------------------------------------------------
# The move library
#
# library/layers.py says how each layer of a move is made:
#   BODY      name -> (function(b, amplitudes, period), shape at k = 1, period in beats)
#   CLAWS     name -> (function(b, intensity, period), period in beats)
#   ACCENTS   name -> function(b, size, body offset, claws) -> body offset
# and optionally BODY_HITS / CLAW_HITS (layers re-timed to hit on given beats of the
# bar, written "rise@0,1"), NEEDS (claw -> the body and size it was made for), FIXED
# (danced at their written size, not measured), UNAVAILABLE (data missing) and
# NOT_SHARED (made from someone else's data).
# library/moves.yaml names the moves: each one a body layer + a claw layer.
# ---------------------------------------------------------------------------

LIBRARY = ROOT / "library"
LAYERS = None       # the library's layers.py, loaded on first use


def use_library(folder=LIBRARY):
    """Load the layers of the library in `folder` (normally library/)."""
    global LAYERS, LIBRARY
    folder = Path(folder) if Path(folder).is_absolute() else ROOT / folder
    if LAYERS is None or folder != LIBRARY:
        if str(folder.parent) not in sys.path:
            sys.path.insert(0, str(folder.parent))
        LAYERS, LIBRARY = importlib.import_module(f"{folder.name}.layers"), folder
    return LAYERS


def layers():
    return LAYERS if LAYERS is not None else use_library()


def named_moves():
    """The library's named moves (library/moves.yaml): name -> its spec."""
    return yaml.safe_load((LIBRARY / "moves.yaml").read_text())["moves"]


def station_from(step):
    """A Station from a choreography step: either a named move from the library
    ({move: P01_bounce_counter, beats: 8}) or the layers spelled out ({body, claw, ...})."""
    step = dict(step)
    if "move" in step:
        spec = {k: v for k, v in named_moves()[step.pop("move")].items() if k not in ("mv", "what")}
        step = {**spec, **step}
    return Station(**dict(step, beats=float(step["beats"])))


@dataclass
class ClawOut:
    """What a claw move returns for one moment."""
    left: np.ndarray            # LF joints J0..J4
    right: np.ndarray           # RF joints, already mirrored
    ground: dict = field(default_factory=dict)   # arm -> 0..1, pull the pad onto the sand
    tip_base: dict | None = None    # palm-pad targets in the base frame
    tip_world: dict | None = None   # palm-pad targets fixed in the world (isolation)
    tip_follow: float = 0.0         # with tip_world: share of the body's motion the tips follow
    tip_center: tuple = (0.0, 0.0)

    def arm_q(self):
        return {"LF": self.left, "RF": self.right}


def body_offset(name, b, scale, period=None, offset=0.0):
    name, positions = split_hits(name)
    fn, shape, p0 = layers().BODY[name]
    amp = {k: v * scale for k, v in shape.items()}
    if positions is not None:
        return layers().BODY_HITS[name](b - offset, amp, 4.0 if period is None else period, positions)
    return fn(b - offset, amp, p0 if period is None else period)


def claw_pose(name, b, intensity, period=None, offset=0.0):
    name, positions = split_hits(name)
    s = float(np.clip(intensity, 0.0, 1.0))
    if positions is not None:
        return layers().CLAW_HITS[name](b - offset, s, 4.0 if period is None else period, positions)
    fn, p0 = layers().CLAWS[name]
    return fn(b - offset, s, p0 if period is None else period)


def load_limits(path=None):
    path = Path(path) if path else LIBRARY / "limits.yaml"
    return (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}


# ---------------------------------------------------------------------------
# Tempo
# ---------------------------------------------------------------------------

@dataclass
class TempoMap:
    """Beat k is at time beat_times[k]; linear in between, extrapolated past the ends."""
    beat_times: np.ndarray

    @classmethod
    def constant(cls, bpm, n_beats, t0=0.0):
        return cls(t0 + np.arange(n_beats + 1) * 60.0 / bpm)

    def scaled(self, tempo_scale):
        """The same song played tempo_scale times faster (what ffmpeg's atempo does)."""
        return TempoMap(self.beat_times / tempo_scale)

    def beat(self, t):
        bt = self.beat_times
        k = np.arange(len(bt), dtype=float)
        t = np.asarray(t, dtype=float)
        out = np.interp(t, bt, k)
        out = np.where(t < bt[0], (t - bt[0]) / (bt[1] - bt[0]), out)
        return np.where(t > bt[-1], k[-1] + (t - bt[-1]) / (bt[-1] - bt[-2]), out)

    def time(self, b):
        bt = self.beat_times
        k = np.arange(len(bt), dtype=float)
        b = np.asarray(b, dtype=float)
        out = np.interp(b, k, bt)
        out = np.where(b < 0, bt[0] + b * (bt[1] - bt[0]), out)
        return np.where(b > k[-1], bt[-1] + (b - k[-1]) * (bt[-1] - bt[-2]), out)

    def bpm_at(self, t):
        bt = self.beat_times
        ibi = np.diff(bt)
        return 60.0 / np.interp(t, bt[:-1] + ibi / 2, ibi)


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

@dataclass
class HoldHome:
    beats: float


@dataclass
class Enter:
    beats: float = 4.0


@dataclass
class Exit:
    beats: float = 4.0


@dataclass
class Station:
    beats: float
    body: str = "still"
    claw: str = "hold_up"
    intensity: float = 0.9          # body amplitude, as a fraction of the largest safe one
    claw_intensity: float = 1.0
    accents: dict = field(default_factory=dict)
    body_period: float | None = None
    body_offset: float = 0.0
    claw_period: float | None = None
    claw_offset: float = 0.0
    blend: float = 1.0              # beats of crossfade from the previous station
    k_override: float | None = None   # fixed amplitude scale, used by the limit probe

    def label(self):
        a = "+".join(self.accents)
        return f"{self.body}/{self.claw}" + (f"/{a}" if a else "")


def _interp_table(table, bpm):
    if isinstance(table, (int, float)):
        return float(table)
    keys = sorted(table, key=float)
    return float(np.interp(bpm, [float(x) for x in keys], [float(table[x]) for x in keys]))


@dataclass
class Frame:
    base_off: np.ndarray       # [dx dy dz roll pitch yaw] from the DANCE stance
    claws: ClawOut
    foot_lift: dict


_warned = set()


class Composer:
    def __init__(self, segments, tempo, limits=None, start_beat=0.0):
        self.segments = segments
        self.tempo = tempo
        self.limits = load_limits() if limits is None else limits
        self.starts = start_beat + np.concatenate([[0.0], np.cumsum([s.beats for s in segments])])
        self.kin = Kin()

    def k_max(self, body, claw, bpm):
        """Largest safe amplitude for this body + claw pair at this tempo (limits.yaml)."""
        body, claw = body.split("@")[0], claw.split("@")[0]
        row = self.limits.get("k_max", {}).get(body, {})
        table = row.get(claw, row.get("*"))
        return 1.0 if table is None else _interp_table(table, bpm)

    def claw_max(self, claw, bpm):
        table = self.limits.get("claw_max", {}).get(claw.split("@")[0])
        return 1.0 if table is None else _interp_table(table, bpm)

    def station(self, s, b):
        bpm = float(self.tempo.bpm_at(self.tempo.time(b)))
        if s.k_override is not None:
            scale = s.k_override
        else:
            k = self.k_max(s.body, s.claw, bpm)
            if k < 0:
                # e.g. a 16th-note dig at 156 BPM is too fast for the servos: keep the
                # body still and let claw_max shrink the stroke instead
                if (s.body, s.claw, round(bpm)) not in _warned:
                    _warned.add((s.body, s.claw, round(bpm)))
                    print(f"[compose] {s.body}/{s.claw} is too fast at {bpm:.0f} BPM; "
                          f"body still, smaller claw stroke", flush=True)
                k = 0.0
            scale = s.intensity * k
            need = getattr(layers(), "NEEDS", {}).get(s.claw.split("@")[0])
            if need and need[0] == s.body.split("@")[0]:
                scale = max(scale, need[1])     # e.g. the dig was made for a full crouch
        off = body_offset(s.body, b, scale, s.body_period, s.body_offset)
        ci = s.claw_intensity if s.k_override is not None else \
            min(s.claw_intensity, self.claw_max(s.claw, bpm))
        claws = claw_pose(s.claw, b, ci, s.claw_period, s.claw_offset)
        for name, a in s.accents.items():
            off = layers().ACCENTS[name](b, a, off, claws)
        return Frame(off, claws, {})

    def _station_at(self, i, b):
        s = self.segments[i]
        f = self.station(s, b)
        u = b - self.starts[i]
        if i > 0 and isinstance(self.segments[i - 1], Station) and u < s.blend:
            f = _mix(self.station(self.segments[i - 1], b), f, smootherstep(u / s.blend))
        return f

    def targets(self, b):
        i = int(np.clip(np.searchsorted(self.starts, b, side="right") - 1, 0, len(self.segments) - 1))
        seg, u = self.segments[i], b - self.starts[i]
        if isinstance(seg, HoldHome):
            return self._home_targets(), "home"
        if isinstance(seg, Station):
            return self._dance_targets(self._station_at(i, b)), seg.label()
        # Enter: first half steps the legs in, second half raises into the next move.
        # Exit is the same backwards.
        half = seg.beats / 2.0
        entering = isinstance(seg, Enter)
        if entering and u < half:
            return self._stepping(u / half, forward=True), "enter:step"
        if not entering and u >= half:
            return self._stepping((u - half) / half, forward=False), "exit:step"
        nxt = self._neighbour_station(i, +1 if entering else -1)
        dance = self.station(nxt, b) if nxt else Frame(np.zeros(6), _home_claws(), {})
        if entering:
            return self._raise(smootherstep((u - half) / half), dance), "enter:raise"
        return self._raise(1.0 - smootherstep(u / half), dance), "exit:lower"

    def _neighbour_station(self, i, step):
        j = i + step
        while 0 <= j < len(self.segments):
            if isinstance(self.segments[j], Station):
                return self.segments[j]
            if not isinstance(self.segments[j], (Enter, Exit)):
                return None
            j += step
        return None

    def _home_targets(self):
        h = HOME_STANCE
        return Targets(base_pos=np.array([0.0, 0.0, h.base_z]), base_quat=np.array([1.0, 0, 0, 0]),
                       feet={s: h.feet[s].copy() for s in SUPPORT},
                       arm_q={a: h.arm_q(a) for a in ARMS},
                       claw_tip={a: h.feet[a].copy() for a in ARMS}, leg_q=h.leg_q)

    def _stepping(self, p, forward):
        """Move the legs between the walking and dancing footholds one at a time, claws
        still planted, so every lift has five points of support."""
        t = self._home_targets()
        order = STEP_ORDER if forward else tuple(reversed(STEP_ORDER))
        for k, leg in enumerate(order):
            q = smootherstep(np.clip(p * len(order) - k, 0.0, 1.0))
            a, z = HOME_STANCE.feet[leg], DANCE.foot(leg)
            if not forward:
                a, z = z, a
            pos = a + (z - a) * q
            pos[2] = DANCE.feet_z + STEP_LIFT * np.sin(np.pi * q)
            t.feet[leg] = pos
        done = smootherstep(p)
        t.leg_q = HOME_STANCE.leg_q + (DANCE.leg_q - HOME_STANCE.leg_q) * (done if forward else 1.0 - done)
        return t

    def _raise(self, r, dance):
        """r = 0: dancing footholds, claws still on the sand. r = 1: the dance frame."""
        t = self._dance_targets(Frame(dance.base_off * r, dance.claws, dance.foot_lift if r > 0.99 else {}))
        h = HOME_STANCE
        z = h.base_z + (DANCE.base_z + dance.base_off[2] - h.base_z) * r
        t.base_pos = np.array([dance.base_off[0] * r, dance.base_off[1] * r, z])
        t.base_quat = quat_from_rpy(*(np.array([dance.base_off[3], DANCE.pitch + dance.base_off[4],
                                                dance.base_off[5]]) * r))
        # Lift the arms first, then turn them in. Moving every joint together swings the
        # two claws across each other in front of the face.
        r_lift = smootherstep(min(1.0, r / 0.7))
        r_yaw = smootherstep(max(0.0, (r - 0.3) / 0.7))
        w = np.array([r_yaw, r_lift, r_lift, r_lift, r_lift])
        for a in ARMS:
            t.arm_q[a] = h.arm_q(a) + (dance.claws.arm_q()[a] - h.arm_q(a)) * w
        t.claw_tip = {a: h.feet[a].copy() for a in ARMS} if r < 0.15 else {}
        return t

    def _dance_targets(self, f):
        base_pos, base_quat = base_pose(f.base_off)
        feet = {}
        for s in SUPPORT:
            p = DANCE.foot(s)
            p[2] += f.foot_lift.get(s, 0.0)
            feet[s] = p
        t = Targets(base_pos=base_pos, base_quat=base_quat, feet=feet,
                    arm_q={a: q.copy() for a, q in f.claws.arm_q().items()}, leg_q=DANCE.leg_q)
        if f.claws.tip_base is not None:
            R = quat_mat(base_quat)
            for a, p in f.claws.tip_base.items():
                t.claw_tip[a] = base_pos + R @ p
        if f.claws.tip_world is not None:
            follow = f.claws.tip_follow * (base_pos[:2] - np.asarray(f.claws.tip_center))
            for a, p in f.claws.tip_world.items():
                q = np.asarray(p, dtype=float).copy()
                q[:2] += follow
                t.claw_tip[a] = q
        for a, w in f.claws.ground.items():
            if w > 0.0:
                p = self._claw_fk(base_pos, base_quat, t.arm_q)[a]
                p[2] = p[2] + (DANCE.feet_z + 0.002 - p[2]) * w
                t.claw_tip[a] = p
        return t

    def _claw_fk(self, base_pos, base_quat, arm_q):
        q = home_qpos()
        q[:3], q[3:7] = base_pos, base_quat
        q[7:12], q[12:17] = arm_q["LF"], arm_q["RF"]
        self.kin.set(q)
        return {a: self.kin.site_pos(a) for a in ARMS}

    def run(self, dt=CONTROL_DT, t_start=None, t_end=None, progress=False):
        t0 = float(self.tempo.time(self.starts[0])) if t_start is None else t_start
        t1 = float(self.tempo.time(self.starts[-1])) if t_end is None else t_end
        times = np.arange(t0, t1, dt)
        ik = WholeBodyIK(DANCE.leg_q)
        ik.reset(home_qpos())
        # Settle on the first frame before stepping through time. Arms go to their
        # keyframes first with the claw targets left out: an arm has an elbow-up and an
        # elbow-down way to reach the same point, and from HOME the targets alone can
        # pull it onto the wrong one (claws dragging in the sand) for the whole clip.
        step = ik.max_joint_step
        ik.max_joint_step = None
        first = self.targets(float(self.tempo.beat(times[0])))[0]
        ik.solve(replace(first, claw_tip={}), iters=400, collisions=False)
        ik.solve(first, iters=400, collisions=False)
        ik.solve(first, iters=100)
        ik.max_joint_step = step
        qs, beats, labels, errs = [], [], [], []
        for n, t in enumerate(times):
            b = float(self.tempo.beat(t))
            tg, lab = self.targets(b)
            q, e = ik.solve(tg)
            qs.append(q)
            beats.append(b)
            labels.append(lab)
            errs.append((e["foot"], max(e["tip"].values(), default=0.0), e["base"]))
            if progress and n % 500 == 0:
                print(f"  frame {n}/{len(times)}  beat {b:.1f}  {lab}", flush=True)
        return Trajectory(times=times - t0, qpos=np.array(qs), beats=np.array(beats), labels=labels,
                          ik_err=np.array(errs), dt=dt, beat_times=self.tempo.beat_times - t0)


def _home_claws():
    return ClawOut(HOME_STANCE.arm_q("LF"), HOME_STANCE.arm_q("RF"))


def _tips_world(f):
    if f.claws.tip_world is not None:
        return f.claws.tip_world
    pos, quat = base_pose(f.base_off)
    R = quat_mat(quat)
    tb = f.claws.tip_base if f.claws.tip_base is not None else arm_tips_base(f.claws.arm_q())
    return {a: pos + R @ p for a, p in tb.items()}


def _mix(a, b, w):
    """Crossfade two frames, w = 0 is all a."""
    ground = {k: a.claws.ground.get(k, 0.0) * (1 - w) + b.claws.ground.get(k, 0.0) * w
              for k in set(a.claws.ground) | set(b.claws.ground)}
    tips, tips_w = None, None
    if a.claws.tip_world is not None or b.claws.tip_world is not None:
        ta, tb = _tips_world(a), _tips_world(b)
        tips_w = {k: ta[k] * (1 - w) + tb[k] * w for k in ta}
    elif a.claws.tip_base is not None or b.claws.tip_base is not None:
        ta = a.claws.tip_base or arm_tips_base(a.claws.arm_q())
        tb = b.claws.tip_base or arm_tips_base(b.claws.arm_q())
        tips = {k: ta[k] * (1 - w) + tb[k] * w for k in ta}
    claws = ClawOut(a.claws.left * (1 - w) + b.claws.left * w,
                    a.claws.right * (1 - w) + b.claws.right * w, ground, tips, tips_w,
                    tip_follow=a.claws.tip_follow * (1 - w) + b.claws.tip_follow * w,
                    tip_center=b.claws.tip_center if b.claws.tip_world is not None else a.claws.tip_center)
    lift = {k: a.foot_lift.get(k, 0.0) * (1 - w) + b.foot_lift.get(k, 0.0) * w
            for k in set(a.foot_lift) | set(b.foot_lift)}
    return Frame(a.base_off * (1 - w) + b.base_off * w, claws, lift)


@dataclass
class Trajectory:
    times: np.ndarray       # (N,) s, from 0
    qpos: np.ndarray        # (N, 29)
    beats: np.ndarray       # (N,)
    labels: list
    ik_err: np.ndarray      # (N, 3) foot, claw tip, base (m)
    dt: float
    beat_times: np.ndarray  # relative to t = 0

    def save(self, path):
        np.savez_compressed(path, times=self.times, qpos=self.qpos, beats=self.beats,
                            labels=np.array(self.labels), ik_err=self.ik_err, dt=self.dt,
                            beat_times=self.beat_times)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["times"], z["qpos"], z["beats"], [str(x) for x in z["labels"]],
                   z["ik_err"], float(z["dt"]), z["beat_times"])


# ---------------------------------------------------------------------------
# Writing a clip the jumper dance task can train on
# ---------------------------------------------------------------------------

def write_clip(traj, path, bpm, source, audio_start_s=0.0):
    """Save in jumper's dance clip format: one channel per base DOF and per joint, legs
    numbered round the body (L0 LF, L1 LM, L2 LR, L3 RR, L4 RM, L5 RF)."""
    from tasks.jumper.common.dance.motion import LEG_JOINTS

    q = traj.qpos
    rpy = np.array([rpy_from_quat(x[3:7]) for x in q])
    rpy[:, 2] = np.unwrap(rpy[:, 2])
    ch = {"body_x": q[:, 0], "body_y": q[:, 1], "body_z": q[:, 2],
          "body_roll": rpy[:, 0], "body_pitch": rpy[:, 1], "body_yaw": rpy[:, 2]}
    for leg, names in LEG_JOINTS.items():
        for j, name in enumerate(names):
            ch[f"L{leg}_j{j}"] = q[:, 7 + JOINTS.index(name)]
    ch = {k: np.asarray(v, dtype=np.float32) for k, v in ch.items()}
    n = len(q)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **ch, **{f"meas_{k}": v for k, v in ch.items()},   # commanded == achieved here
        time=(np.arange(n) * traj.dt).astype(np.float32), dt=np.float64(traj.dt),
        phase=np.ones(n, dtype=np.float32), bpm=np.float64(bpm),
        beat_times_sim=np.asarray(traj.beat_times, dtype=np.float64),
        audio_start_in_sim=np.float64(audio_start_s), source=np.array(source))
    return path


def validate_clip(folder, control_dt=CONTROL_DT):
    """Run jumper's own converter on a folder holding one clip. It checks joint limits,
    flat feet and continuity, and raises with its own message if anything is off."""
    from tasks.jumper.common.dance.motion import ensure_motion_npz

    return ensure_motion_npz(control_dt, Path(folder))


# ---------------------------------------------------------------------------
# Choreography files
#
#   music: examples/my_song.mp3
#   bpm: 128                          # the song's tempo (or `beats:` a file of beat times)
#   first_beat: 0.42                  # seconds into the song
#   start_beat: 16                    # when the dance starts, in beats
#   sections:
#     - walk: {move: shuffle, beats: 8}
#     - dance:
#         - {move: P01_bounce_counter, beats: 8}              # a move from the library
#         - {move: P24_dig, beats: 8, intensity: 0.8}         # smaller than usual
#         - {body: swing_y, claw: alternate, beats: 8}        # or any body + claw layers
#
# A `dance` block gets an Enter before and an Exit after, unless it says
# `dance: {moves: [...], enter: false, exit: false}`.
# ---------------------------------------------------------------------------

@dataclass
class WalkSection:
    start: float
    beats: float
    move: str
    intensity: float = 0.9
    period: float = 4.0


@dataclass
class DanceBlock:
    start: float
    beats: float
    segments: list


@dataclass
class Show:
    tempo: TempoMap
    start_beat: float
    sections: list
    music: Path | None = None

    @property
    def end_beat(self):
        s = self.sections[-1]
        return s.start + s.beats

    def scaled(self, tempo_scale):
        return Show(self.tempo.scaled(tempo_scale), self.start_beat, self.sections, self.music)

    def section_at(self, b):
        return max(i for i, s in enumerate(self.sections) if s.start <= b + 1e-9)


def song_tempo(d):
    """The beat grid of a choreography: a file of beat times, or `bpm` + `first_beat`."""
    if "beats" in d:
        return TempoMap(np.load(ROOT / d["beats"]))
    return TempoMap.constant(float(d["bpm"]), 4000, float(d.get("first_beat", 0.0)))


def load_show(src):
    d = src if isinstance(src, dict) else yaml.safe_load(Path(src).read_text())
    if "library" in d:
        use_library(d["library"])
    b = start = float(d.get("start_beat", 0))
    sections = []
    for sec in d["sections"]:
        if "walk" in sec:
            w = sec["walk"]
            sections.append(WalkSection(b, float(w["beats"]), w["move"],
                                        float(w.get("intensity", 0.9)), float(w.get("period", 4.0))))
        else:
            spec = sec["dance"]
            steps = spec["moves"] if isinstance(spec, dict) else spec
            opts = spec if isinstance(spec, dict) else {}
            segs = [Enter()] if opts.get("enter", True) else []
            segs += [station_from(st) for st in steps]
            if opts.get("exit", True):
                segs.append(Exit())
            sections.append(DanceBlock(b, sum(s.beats for s in segs), segs))
        b += sections[-1].beats
    music = ROOT / d["music"] if d.get("music") else None
    return Show(song_tempo(d), start, sections, music)
