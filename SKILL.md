---
name: jumper-reference-motion
description: Turn a description of a dance move (in words, or a moment in a video) into a reference motion the Jumper robot crab can dance, add it to the move library, and train the policy to follow it. Also covers dancing the library to a new song. Use with this repository.
---

# Making reference motions for Jumper

A reference motion is the motion the robot should do, frame by frame: where the body is
and what all 22 joints are, 50 times a second. It's pure geometry, no physics. The
policy (`checkpoint/policy.pt`) is what then makes the simulated robot follow it.

The path for a new move:

    description -> layers.py -> moves.yaml -> limits -> look at it -> references -> train

Everything lives in `library/`:

- `layers.py`: how each layer moves. A move is a **body layer** (how the shell moves over
  the four planted legs) plus a **claw layer** (what the two arms do), sometimes with an
  **accent** (a pincer snap, a dip on the beat).
- `moves.yaml`: the named moves, each a body layer + a claw layer.
- `limits.yaml`: how big each body + claw pair can be on this robot.

## 1. Pin the description down

From the words or the video, work out:

- which part moves: the body, the claws, or both; any small extra on the beat
- the rhythm: every beat, every 2 beats, or on given beats of the bar (1, 2, 3, 3-and, 4)
- the strongest moment (the clap, the lowest point, the hit): it goes exactly on the beat
- the feel: a smooth back and forth, a hit and release, a dip, a sequence of poses
- roughly how big (cm, degrees), and what must stay fixed (do the feet stay down?)

Say it back in a sentence or two and agree on it before writing code. Words are often
ambiguous: a "jump" may really mean "rise up, feet on the ground".

## 2. Write the layers in library/layers.py

Reuse existing layers where you can; write new ones where you can't.

- A body layer is `f(b, a, period)`, returning the offset `[dx, dy, dz, roll, pitch, yaw]`
  from the dancing stance (m and rad; roll + is left side up, pitch + is nose down). `a`
  holds its amplitudes. Add it to `BODY` with its amplitudes at size 1 and its period in
  beats.
- A claw layer is `f(b, s, period)`, returning `ClawOut(left, mirror(right))`, usually a
  blend between keyframes (joint angles of the left arm) driven by a curve. `s` (0 to 1)
  is how big. Add it to `CLAWS`.
- `b` is the beat count as a float: beat 3 starts at `b = 3.0`. That's why every move works
  at any tempo. Build moves from the curves in `motion/curves.py`:
  - `swing` for smooth sways
  - `strike` for hits and claps
  - `hits` for irregular rhythms
  - `catmull_rom_loop` for a sequence of poses
- Keyframes that only need to look right: pick them by eye. Keyframes that must touch
  something (the sand, the other claw): solve them, like `library/solve_dig.py`.
- Examples to copy:
  - `_rise` with `_rise_clap`: stand up and clap above the head every 2 beats
  - `_swing_y` with `_swing_side`: a side to side sway with the claws following
  - `_crouch` with `_dig`: four poses in a loop, raking the sand

What makes moves read well:
- The extreme lands exactly on a beat.
- Every move keeps moving; no frozen holds.
- In a group, offsets between dancers are whole beats, never half a beat. Half a beat looks
  late, not like a ripple.

## 3. Name it in library/moves.yaml

    P30_my_move: {body: my_body, claw: my_claw, mv: "1:26.4", what: "one line describing it"}

Optional fields:
- `accents: {snap: 1.0}`
- `body_offset: 1.0` and `claw_offset: 1.0` (start a beat later)
- `intensity` (0 to 1, default 0.9)

## 4. Measure how big it can be

    python motion/limits.py --bodies my_body --claws my_claw

This tries the move bigger and bigger at 90, 125 and 160 BPM, and keeps the largest size
that still passes four checks:
- every joint angle is reachable
- the motors are fast enough
- the centre of mass stays over the feet
- the claws stay clear of the shell and of each other

The result goes into `library/limits.yaml`. `-1` means it can't be danced at that tempo at
all: make it smaller or pair it differently.

## 5. Look at it

Write a short choreography with the move (copy `examples/my_song.yaml`), then:

    python perform/dance.py examples/my_test.yaml --reference

This renders the reference motion, with no physics. Look at it, change the numbers in
step 2, and repeat. This loop is where a move gets its character, and each try takes
seconds.

## 6. Write the reference files

    python motion/make_references.py

This writes one file per named move into `references/`. Each `.npz` holds:
- `qpos`: frames x 29 (base position, base quaternion w x y z, 22 joint angles)
- `joint_names`, `fps`, `beats`, `spec`

## 7. Train the policy to follow it

    python training/make_clip.py
    python training/train.py --init checkpoint/policy.pt --iterations 5000

`training/make_clip.py` builds a long training clip: every move in `moves.yaml` at least once, at
random sizes, while the tempo wanders between 80 and 170 BPM. `training/train.py` teaches the policy
to follow it with PPO.

Starting from the existing policy is the quick way after adding moves. To train from
Jumper's own dance policy instead, run `python training/train.py` (about 20,000 iterations, ~4 hours
on an RTX 4090). Then check the result with physics:

    python perform/dance.py examples/my_test.yaml --policy logs/<run>/model_<last>.pt

If a move comes out small or mushy:
- give it more weight in `library/clip.yaml`
- train longer
- or tighten the tracking widths in `training/env.py`

## Dancing to a new song

The policy follows whatever reference it's given, so the moves work with any song:

1. `python perform/beats.py my_song.mp3` prints the song's `bpm` and `first_beat`.
2. Copy `examples/my_song.yaml`, fill those in, and list moves by name:
   `{move: P20_clap_bounce_snap, beats: 8}`.
3. `python perform/dance.py examples/my_song.yaml` (add `--tempo 1.2` to speed the song up).
