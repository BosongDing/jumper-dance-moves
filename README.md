# Crab Rave moves for a robot crab

A library of 29 dance moves from Noisestorm's *Crab Rave* music video, made for
[Jumper](https://github.com/KingKongRobotics/jumper), KingKong Robotics' 22-joint robot crab
(two 5-joint claw arms, four 3-joint legs). It comes with:
- a trained policy that dances all of them in physics, to any song, at any tempo
- the code to make your own moves and train the policy on them

Everything runs in simulation (MuJoCo).

## What's here

| | |
|---|---|
| `references/` | the moves as reference motions, one file each: what the robot should do, frame by frame |
| `checkpoint/policy.pt` | the trained policy: it makes the simulated robot follow any of them |
| `library/` | how the moves are made: `moves.yaml` (the 29 moves, named), `layers.py` (their body and claw layers), `limits.yaml` (how big each can get) |
| `SKILL.md` | how to make reference motions for new moves, step by step |
| `examples/` | a choreography for Crab Rave, and a template for your own song |
| `motion/` | code that makes reference motions: `curves.py` (beat-shaped curves), `robot.py` (Jumper's body and the inverse kinematics), `compose.py` (moves on the beat -> joint angles), `limits.py`, `make_references.py` |
| `training/` | PPO training: `make_clip.py` (the training clip), `env.py` (what the policy sees and is paid for), `train.py` |
| `perform/` | dancing: `dance.py` (a choreography with physics and the song), `beats.py` (a song's tempo) |

## How it works

**A reference motion** is a move written as curves of the beat count: "at beat 4.5 the
body is here, the claws are like this". Speed the song up and the beat count just grows
faster, so every move follows any tempo. Whole-body inverse kinematics turns that into the
robot's 22 joint angles, 50 times a second, with the feet planted and the claws clear of
the shell. It's pure geometry: a perfect puppet that ignores gravity.

**The policy** is a small neural network trained with PPO (Proximal Policy Optimization)
on 4096 simulated robots at once.
- What it sees, every 1/50 s: where the reference wants the joints now and 0.04, 0.1 and
  0.2 s ahead, plus the robot's own sensors.
- What it does: sets the 22 motor targets.
- What it's rewarded for: matching the reference while staying upright, smooth and easy on
  the motors.

It never learns which move it's doing. It learns to follow whatever reference it's shown,
like a singer reading lyrics off a screen. So one policy dances every move, and any new
choreography of them, to any song. Which move it dances is decided by the reference you
give it.

## Setup

Linux, Python 3.11, conda. A GPU is only needed for training.

```bash
conda create -n crab python=3.11 && conda activate crab
git clone https://github.com/KingKongRobotics/jumper.git
cd jumper && git checkout b2e5866 && pip install -e . && cd ..
pip install -r requirements.txt
```

Jumper brings MuJoCo, mjlab, rsl_rl and torch (the CUDA build). On a machine without an
NVIDIA GPU, install the much smaller CPU build before `pip install -e .`:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

## Dance

```bash
python perform/dance.py examples/crab_rave.yaml                # -> out/crab_rave/show.mp4
python perform/dance.py examples/crab_rave.yaml --tempo 1.25   # the same dance, 25% faster
python perform/dance.py examples/crab_rave.yaml --reference    # the reference motion only, no physics
```

`examples/crab_rave.yaml` needs your own copy of the song as `examples/crab_rave.mp3`.
Without it, the video is silent. One move (P26) is retargeted from GANimator's crab dance,
which isn't ours to share. `python library/ganimator.py` downloads and converts it.

### Your own song

```bash
python perform/beats.py examples/my_song.mp3      # prints bpm and first_beat
```

Put those two numbers in a copy of `examples/my_song.yaml` and list moves by name, like
`{move: P20_clap_bounce_snap, beats: 8}`. Then:

```bash
python perform/dance.py examples/my_song.yaml
```

## The reference files

One `.npz` per move in `references/`, 8 beats at 125 BPM:

| Field | Contents |
|---|---|
| `qpos` | frames x 29: base position (m), base quaternion (w x y z), 22 joint angles (rad) |
| `joint_names` | the 22 joint names |
| `fps` | 50 |
| `beats` | the beat count at each frame |
| `spec` | the move as written in `library/moves.yaml` |

They're plain arrays, so you can use them with other training code too.

## New moves and training

[`SKILL.md`](SKILL.md) walks through it: describe the move, write its layers, measure how
big it can be, look at it, write the references, train. The training part is:

```bash
python training/make_clip.py                                          # every move at every tempo, ~7 min of dance
python training/train.py --init checkpoint/policy.pt --iterations 5000   # after adding moves
python training/train.py                                              # or from scratch: ~4 h on an RTX 4090
```

`policy.pt` was trained in four stages while the moves were being made, about 21,500
PPO iterations in all. It started from Jumper's own dance policy. Each stage's clip was a
random mix of the body and claw layers in `library/layers.py`. `training/make_clip.py` now builds
the clip from the named moves instead, every one at least once, so whatever is in
`moves.yaml` is what a new policy learns.

## Credits

- [Jumper](https://github.com/KingKongRobotics/jumper) (KingKong Robotics): the robot, its
  model, its walking policy, and the dance task this builds on
- [mjlab](https://github.com/mujocolab/mjlab), [rsl_rl](https://github.com/leggedrobotics/rsl_rl),
  [MuJoCo](https://github.com/google-deepmind/mujoco) and [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)
- [BeyondMimic](https://beyondmimic.github.io/) for the motion tracking recipe
- [mink](https://github.com/kevinzakka/mink) for the inverse kinematics
- [GANimator](https://github.com/PeizhuoLi/ganimator) (Li et al., SIGGRAPH 2022) for the crab dance in P26
- Noisestorm for *Crab Rave*. The song isn't in this repo.

## License

Free to use, change and share for noncommercial purposes (personal projects, research,
teaching, hobby and nonprofit use) under the
[PolyForm Noncommercial License 1.0.0](LICENSE.md). Commercial use is not allowed. A few
parts come from Jumper and keep its Apache 2.0 license, see `NOTICE`.
