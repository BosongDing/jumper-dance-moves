"""Measure how big every move of the library can be, and write library/limits.yaml.

For each body + claw layer pair at 90, 125 and 160 BPM: compose 8 beats of it at a size k
(k times the move's written shape), solve the joint angles, and check every frame:
reachable, motors fast enough, centre of mass over the feet, claws clear of the shell
and of each other. Then search for the largest k that passes (halving the range each
try, like guessing a number). The same for each claw move's stroke size, body still.
No physics here: it's all geometry on the planned motion.

Needed whenever you add or change moves. The full table takes a few minutes on 24 cores.

    python limits.py                                    # every pair of layers
    python limits.py --bodies arch --claws clap pump    # just these, merged in
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import yaml

import compose
from compose import Composer, Station, TempoMap, layers, use_library
from robot import evaluate

BEATS = 8
def check(body, claw, bpm, k=0.0, claw_intensity=1.0):
    seg = Station(BEATS + 1, body, claw, k_override=k, claw_intensity=claw_intensity, blend=0.0)
    tr = Composer([seg], TempoMap.constant(bpm, BEATS + 8), limits={}).run()
    skip = int(round(60.0 / bpm / tr.dt))       # the first beat is the IK settling in
    return evaluate(tr.qpos, tr.dt, tr.ik_err, mesh_every=2).violations(skip)


def _init(folder):
    use_library(folder)


def body_limit(job):
    body, claw, bpm = job
    need = getattr(layers(), "NEEDS", {}).get(claw)
    k0 = need[1] if need and need[0] == body else 0.0
    if check(body, claw, bpm, k0):
        return body, claw, bpm, -1.0         # not danceable even at rest
    lo, hi = k0, 1.6
    if not check(body, claw, bpm, hi):
        return body, claw, bpm, hi
    for _ in range(6):
        mid = 0.5 * (lo + hi)
        if check(body, claw, bpm, mid):
            hi = mid
        else:
            lo = mid
    return body, claw, bpm, round(lo, 3)


def claw_limit(job):
    claw, bpm = job
    need = getattr(layers(), "NEEDS", {}).get(claw)
    body, k = need if need else ("still", 0.0)
    if not check(body, claw, bpm, k, 1.0):
        return claw, bpm, 1.0
    lo, hi = 0.0, 1.0
    for _ in range(6):
        mid = 0.5 * (lo + hi)
        if check(body, claw, bpm, k, mid):
            hi = mid
        else:
            lo = mid
    return claw, bpm, round(lo, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies", nargs="*")
    ap.add_argument("--claws", nargs="*")
    ap.add_argument("--bpm", type=float, nargs="*", default=[90.0, 125.0, 160.0])
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    moves = layers()
    path = compose.LIBRARY / "limits.yaml"
    skip = set(getattr(moves, "FIXED", ())) | set(getattr(moves, "UNAVAILABLE", ()))
    partial = bool(a.bodies or a.claws)
    bodies = a.bodies or [b for b in moves.BODY if b != "still" and b not in skip]
    claws = a.claws or [c for c in moves.CLAWS if c not in skip]
    table = yaml.safe_load(path.read_text()) if partial else {"k_max": {}, "claw_max": {}}

    t0 = time.time()
    with Pool(a.jobs, initializer=_init, initargs=(str(compose.LIBRARY),)) as pool:
        for c, bpm, x in pool.imap_unordered(claw_limit, [(c, b) for c in claws for b in a.bpm]):
            table["claw_max"].setdefault(c, {})[bpm] = x
            print(f"claw {c:14s} {bpm:4.0f} BPM  up to {x:.3f}  [{time.time() - t0:.0f} s]", flush=True)
        jobs = [(b, c, bpm) for b in bodies for c in claws for bpm in a.bpm]
        for b, c, bpm, k in pool.imap_unordered(body_limit, jobs):
            table["k_max"].setdefault(b, {}).setdefault(c, {})[bpm] = k
            print(f"{b:12s} {c:14s} {bpm:4.0f} BPM  k_max {k:.3f}  [{time.time() - t0:.0f} s]", flush=True)
    for row in [table["claw_max"]] + list(table["k_max"].values()):
        for name, t in row.items():
            row[name] = dict(sorted(t.items()))

    header = "".join(line + "\n" for line in path.read_text().splitlines()
                     if line.startswith("#")) if path.is_file() else ""
    path.write_text(header + yaml.safe_dump(table, sort_keys=False, default_flow_style=None, width=120))
    print(f"wrote {path} [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
