"""Write one reference motion file per move of the library, into references/.

The moves are the named ones in library/moves.yaml (each a body layer plus a claw layer,
maybe with an accent or an offset). Each is composed for 8 beats at 125 BPM, at 0.9 of
its largest safe size unless it says otherwise, in the dancing stance. You can look at
them, play them, or train something else on them.

One .npz per move, named after it (e.g. P01_bounce_counter.npz):
  qpos         (frames, 29) base position (m), base quaternion (w x y z), then the
               22 joints in joint_names order (rad)
  joint_names  the 22 joint names
  fps          50
  beats        the beat count at each frame
  spec         the move as written in moves.yaml (json), and bpm

    python make_references.py
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import compose
from compose import CONTROL_DT, Composer, Station, TempoMap, layers, load_limits, named_moves, use_library
from robot import JOINTS

ROOT = Path(__file__).resolve().parent

BPM, BEATS, INTENSITY = 125.0, 8, 0.9


def write_one(job):
    folder, name, spec = job
    use_library(folder)
    move = {k: v for k, v in spec.items() if k not in ("mv", "what")}
    move.setdefault("intensity", INTENSITY)
    tempo = TempoMap.constant(BPM, BEATS + 8)
    tr = Composer([Station(BEATS, blend=0.0, **move)], tempo, load_limits()).run()
    path = ROOT / "references" / f"{name}.npz"
    np.savez_compressed(path, qpos=tr.qpos.astype(np.float32), joint_names=np.array(JOINTS),
                        fps=np.float64(1.0 / CONTROL_DT), beats=tr.beats.astype(np.float32),
                        spec=json.dumps(spec), bpm=BPM)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    held_back = set(getattr(layers(), "NOT_SHARED", ()))
    jobs = [(str(compose.LIBRARY), name, spec) for name, spec in named_moves().items()
            if not held_back & {spec["body"], spec["claw"]}]
    (ROOT / "references").mkdir(exist_ok=True)
    with Pool(a.jobs) as pool:
        for path in pool.imap_unordered(write_one, jobs):
            print(f"wrote {path.name}", flush=True)
    print(f"{len(jobs)} references in {ROOT / 'references'}")


if __name__ == "__main__":
    main()
