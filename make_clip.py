"""Build the reference clip the policy trains on.

A long random dance made of the library's moves (library/moves.yaml): every move at least
once, then random ones, a few beats each, at random sizes, stepping in and out of the
dancing stance now and then, while the tempo wanders (slow drifts, ramps, sudden jumps on
a bar line). The policy always sees the reference, so if the clip has every move and every
tempo in it, it learns to follow whatever you give it later. Settings: library/clip.yaml.

Every frame is checked for balance, reach and speed. A move that fails is danced
smaller and the clip rebuilt, a few times if needed.

    python make_clip.py                 # ~6 min, writes clip/
    python make_clip.py --minutes 1     # quick look (still every move once)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

import compose
from compose import Composer, Enter, Exit, HoldHome, Station, TempoMap, layers, load_limits, named_moves, \
    validate_clip, write_clip
from robot import THRESH, evaluate

ROOT = Path(__file__).resolve().parent


def tempo_curve(rng, n_beats, lo, hi):
    """BPM for each beat: held tempos joined by ramps, jumps and slow drifts."""
    bpm = np.empty(n_beats)
    cur = rng.uniform(110, 135)
    i = 0
    while i < n_beats:
        hold = int(rng.choice([8, 12, 16, 24, 32]))
        target = float(np.clip(rng.normal(125, 25), lo, hi)) if rng.random() < 0.8 else \
            float(rng.uniform(lo, hi))
        kind = rng.choice(["ramp", "jump", "drift"], p=[0.55, 0.25, 0.20])
        if kind == "jump":
            i0 = i + (4 - i % 4) % 4           # on a bar line
            bpm[i:i0] = cur
            i = i0
            cur = target
            ramp = np.full(hold, cur)
        elif kind == "ramp":
            n = int(rng.choice([4, 8]))
            ramp = np.concatenate([np.linspace(cur, target, n), np.full(hold, target)])
            cur = target
        else:
            ramp = np.linspace(cur, target, hold + 16)
            cur = target
        j = min(n_beats, i + len(ramp))
        bpm[i:j] = ramp[: j - i]
        i = j
    return bpm


def random_station(rng, cfg, moves, name=None, prev=None):
    """One move for a few beats at a random size: `name`, or a random one."""
    if name is None:
        names = list(moves)
        w = np.array([cfg["weights"].get(n, 1.0) * (0.15 if n == prev else 1.0) for n in names])
        name = names[int(rng.choice(len(names), p=w / w.sum()))]
    spec = {k: v for k, v in moves[name].items() if k not in ("mv", "what", "intensity")}
    st = Station(beats=float(rng.choice(cfg["lengths"])), intensity=float(rng.uniform(*cfg["intensity"])),
                 claw_intensity=float(rng.choice([1.0, 1.0, 1.0, rng.uniform(*cfg["claw_intensity"])])),
                 **spec)
    st.name = name
    return st


def step_beats(bpm, at):
    # Stepping in / out moves one leg at a time in half the segment. Over 140 BPM four
    # beats is too quick for the legs, so take eight.
    i = int(np.clip(at, 0, len(bpm) - 1))
    return 8.0 if float(np.max(bpm[i:i + 8])) > 140.0 else 4.0


def build_segments(rng, cfg, moves, n_beats, bpm):
    first = list(moves)
    rng.shuffle(first)                          # every move at least once, in random order
    segs = [HoldHome(2), Enter(step_beats(bpm, 2))]
    total, prev, since_exit = 2.0 + segs[-1].beats, None, 0
    while total < n_beats - 8 or first:
        if since_exit >= rng.integers(6, 11):      # back to the walking stance now and then
            ex = Exit(step_beats(bpm, total))
            hold = HoldHome(float(rng.choice([1, 2, 4])))
            en = Enter(step_beats(bpm, total + ex.beats + hold.beats))
            segs += [ex, hold, en]
            total += ex.beats + hold.beats + en.beats
            since_exit, prev = 0, None
            continue
        st = random_station(rng, cfg, moves, first.pop() if first else None, prev)
        segs.append(st)
        total += st.beats
        prev, since_exit = st.name, since_exit + 1
    segs += [Exit(step_beats(bpm, total)), HoldHome(2)]
    return segs


def failures(traj, comp):
    """Which segment each bad frame is in, and why."""
    rep = evaluate(traj.qpos, traj.dt, traj.ik_err, mesh_every=2)
    stepping = np.array([lab.endswith(":step") for lab in traj.labels])
    checks = {
        "speed": np.r_[rep.speed > THRESH["speed"], False],
        "com": rep.com < THRESH["com"],
        "zmp": rep.zmp < THRESH["zmp"],
        "clearance": np.nan_to_num(rep.clearance, nan=1.0) < THRESH["clearance"],
        "gap": np.nan_to_num(rep.gap, nan=1.0) < THRESH["gap"],
        "foot": traj.ik_err[:, 0] > THRESH["foot_err"],
        # while a leg steps at a fast tempo the base drifts up to ~15 mm, that's fine
        "base": traj.ik_err[:, 2] > np.where(stepping, 0.02, THRESH["base_err"]),
    }
    seg = np.clip(np.searchsorted(comp.starts, traj.beats, side="right") - 1, 0, len(comp.segments) - 1)
    bad = {}
    for name, mask in checks.items():
        for i in sorted(set(seg[mask].tolist())):
            bad.setdefault(i, {})[name] = int((mask & (seg == i)).sum())
    return dict(sorted(bad.items())), rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--repairs", type=int, default=6)
    a = ap.parse_args()

    cfg = yaml.safe_load((compose.LIBRARY / "clip.yaml").read_text())
    minutes = a.minutes or cfg["minutes"]
    seed = cfg["seed"] if a.seed is None else a.seed
    limits = load_limits()
    missing = set(getattr(layers(), "UNAVAILABLE", ()))
    moves = {n: m for n, m in named_moves().items() if not missing & {m["body"], m["claw"]}}
    if missing:
        print(f"leaving out the moves made with {', '.join(missing)}: data missing (see library/layers.py)")

    rng = np.random.default_rng(seed)
    n_beats = int(minutes * 125)
    bpm = tempo_curve(rng, n_beats + 64, *cfg["bpm"])
    tempo = TempoMap(np.concatenate([[0.0], np.cumsum(60.0 / bpm)]))
    segs = build_segments(rng, cfg, moves, n_beats, bpm)
    print(f"{len(moves)} moves, {len(segs)} segments, {bpm.min():.0f}-{bpm.max():.0f} BPM")

    safe_spec = moves[cfg["safe"]]
    safe = (safe_spec["body"], safe_spec["claw"])
    strikes = {}
    for attempt in range(a.repairs + 1):
        t0 = time.time()
        comp = Composer(segs, tempo, limits)
        traj = comp.run()
        bad, rep = failures(traj, comp)
        for i, why in bad.items():
            name = segs[i].label() if isinstance(segs[i], Station) else type(segs[i]).__name__
            print(f"    segment {i:3d} {name:32s} {why}")
        print(f"attempt {attempt}: {traj.times[-1]:.0f} s of dance in {time.time() - t0:.0f} s, "
              f"{len(bad)} segments to fix", flush=True)
        if not bad:
            break
        if attempt == a.repairs:
            raise SystemExit(f"still failing after {a.repairs} repairs: segments {sorted(bad)}")
        for i in bad:
            strikes[i] = strikes.get(i, 0) + 1
            if strikes[i] >= 3:
                # Smaller didn't help, so the trouble is probably the blend from a
                # neighbour. Swap it and its neighbours for something easy.
                for j in (i - 1, i, i + 1):
                    if 0 <= j < len(segs) and isinstance(segs[j], Station) and (segs[j].body, segs[j].claw) != safe:
                        segs[j].body, segs[j].claw, segs[j].accents = safe[0], safe[1], {}
                        segs[j].body_offset = segs[j].claw_offset = 0.0
                        segs[j].name = cfg["safe"]
                        segs[j].intensity = min(segs[j].intensity, 0.6)
                        print(f"  segment {j}: replaced by {cfg['safe']}")
            elif isinstance(segs[i], Station):
                s = segs[i]
                s.intensity *= 0.8
                s.claw_intensity = max(0.5, s.claw_intensity * 0.9)
                s.accents = {k: v * 0.7 for k, v in s.accents.items()}
            else:     # a step in/out failing comes from the pose next to it
                for j in (i - 1, i + 1):
                    if 0 <= j < len(segs) and isinstance(segs[j], Station):
                        segs[j].intensity *= 0.8

    # env.py sets its reward widths and height limit from these
    q = traj.qpos[:, 7:]
    qd = np.diff(q, axis=0) / traj.dt
    stats = dict(seconds=float(traj.times[-1]),
                 joint_pos_rms=float(np.sqrt(np.mean((q - q.mean(axis=0)) ** 2))),
                 joint_vel_rms=float(np.sqrt(np.mean(qd ** 2))),
                 root_z=[float(traj.qpos[:, 2].min()), float(traj.qpos[:, 2].max())],
                 bpm=[float(bpm.min()), float(bpm.max())],
                 checks={k: round(v, 4) for k, v in rep.summary().items()})
    print(json.dumps(stats))

    out = ROOT / "clip"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.npz"):
        f.unlink()
    path = write_clip(traj, out / "training_clip.npz", bpm=float(bpm.mean()),
                      source=f"make_clip.py seed={seed} minutes={minutes}")
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    (out / "segments.json").write_text(json.dumps(
        [dict(kind=type(s).__name__, move=getattr(s, "name", None), **asdict(s)) for s in segs], indent=1))
    validate_clip(out)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
