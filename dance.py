"""Dance a choreography in MuJoCo with physics, and put the music on it.

    python dance.py examples/crab_rave.yaml                # -> out/crab_rave/show.mp4
    python dance.py examples/crab_rave.yaml --tempo 1.25   # same dance, song 25% faster
    python dance.py examples/crab_rave.yaml --reference    # just the planned motion, no physics

Dance blocks are performed by our policy, tracking a reference composed on the song's
own beat grid. Walk sections use the walking (posture) policy that ships with jumper,
fed beat-timed commands. When the robot switches between the two, its full state is
handed over, so there's no jump.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import warnings
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore", message=".*torch.jit.script.*")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import tasks  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from compose import CONTROL_DT, Composer, DanceBlock, HoldHome, Trajectory, WalkSection, load_limits, \
    load_show, validate_clip, write_clip  # noqa: E402
from curves import pulse  # noqa: E402
from robot import JOINTS, home_qpos, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent
DANCE_POLICY = ROOT / "checkpoint" / "policy.pt"
WALK_POLICY = Path(tasks.__file__).parent / "jumper/posture/out/example/model_74800.pt"
DT = CONTROL_DT


# ---------------------------------------------------------------------------
# Walking: commands for jumper's walking (posture) policy
#
# It takes twist [vx, vy, wz] (m/s, m/s, rad/s) and posture [twist, pitch, roll,
# height]. Measured on that policy: it lags the command by 65-155 ms and halves a
# bounce at one per beat, so every channel is sent early by its lag and the bounce is
# sent 1.8x bigger.
# ---------------------------------------------------------------------------

V_MAX = 0.8          # m/s, what it was trained to
WZ_MAX = 4.0         # rad/s
ROLL_WALK = 0.26
PITCH_WALK = 0.26
STAND_H = 0.10647    # m, normal height
BOUNCE_H = 0.025
H_RANGE = (0.07, 0.15)
LEAD_S = {"vel": 0.066, "roll": 0.10, "height": 0.086}
BOUNCE_GAIN = 1.8

WALKS = ("shuffle", "sway_walk", "march", "spin", "stand")


def walk_command(move, b, bpm, s=0.9, period=4.0):
    """shuffle: side to side (MV 1:04), march: forward / back (MV 0:59), spin: turn in
    place (MV 1:15), sway_walk: shuffle with a roll. Direction flips every half period."""
    s = float(np.clip(s, 0.0, 1.0))
    beat_s = 60.0 / bpm
    bh = b + LEAD_S["height"] / beat_s
    bv = b + LEAD_S["vel"] / beat_s
    br = b + LEAD_S["roll"] / beat_s
    dip = BOUNCE_GAIN * BOUNCE_H * s * pulse(bh, 1.0, width=0.6)
    h = float(np.clip(STAND_H - dip, *H_RANGE))
    sq = float(np.tanh(3.0 * np.sin(2.0 * np.pi * bv / period)))
    sq_r = float(np.tanh(3.0 * np.sin(2.0 * np.pi * br / period)))
    if move == "shuffle":
        return {"twist": [0.0, V_MAX * s * sq, 0.0], "posture": [0.0, 0.0, 0.0, h]}
    if move == "sway_walk":
        return {"twist": [0.0, V_MAX * s * sq, 0.0], "posture": [0.0, 0.0, -ROLL_WALK * s * sq_r, h]}
    if move == "march":
        return {"twist": [V_MAX * s * sq, 0.0, 0.0], "posture": [0.0, PITCH_WALK * 0.5 * s * sq, 0.0, h]}
    if move == "spin":
        bar_s = period * 60.0 / bpm
        wz = min(2.0 * np.pi / bar_s, WZ_MAX) * s
        direction = 1.0 if int(np.floor(b / (2 * period))) % 2 == 0 else -1.0
        u = (b % (2 * period)) / (2 * period)
        env = float(np.clip(np.sin(np.pi * u) * 3.0, 0.0, 1.0))   # ease in and out of each turn
        return {"twist": [0.0, 0.0, direction * wz * env], "posture": [0.0, 0.0, 0.0, h]}
    if move == "stand":
        return {"twist": [0.0, 0.0, 0.0], "posture": [0.0, 0.0, 0.0, STAND_H]}
    raise KeyError(move)


# ---------------------------------------------------------------------------
# Policies in the simulator
# ---------------------------------------------------------------------------

def _cpu_backend():
    from mjrl.backend import resolve
    from mjrl.backend.select import use_backend

    if not getattr(_cpu_backend, "done", False):
        use_backend(resolve(backend="native", device="cpu", num_envs=1, cpu_threads=None, training=False))
        _cpu_backend.done = True


class Robot:
    """One robot in a play-mode environment, driven by one policy."""

    def __init__(self, cfg, agent, runner_cls, checkpoint):
        import torch  # noqa: F401
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper

        _cpu_backend()
        cfg.scene.num_envs = 1
        # the show decides what counts as a fall; an episode reset would teleport it
        cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}
        cfg.episode_length_s = 1e9
        cfg.events.pop("push_robot", None)
        self.env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
        self.wrapped = RslRlVecEnvWrapper(self.env, clip_actions=agent.clip_actions)
        runner = runner_cls(self.wrapped, asdict(agent), device="cpu")
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu")
        self.policy = runner.get_inference_policy(device="cpu")
        self.robot = self.env.scene["robot"]
        self.step_dt = self.env.step_dt
        names = list(self.robot.joint_names)
        self.to_joints = np.array([names.index(j) for j in JOINTS])

    def state(self):
        d = self.robot.data
        return {"pose": d.root_link_pose_w.clone(), "vel": d.root_link_vel_w.clone(),
                "q": d.joint_pos.clone(), "qd": d.joint_vel.clone()}

    def set_state(self, s):
        import torch

        with torch.inference_mode():
            r = self.robot
            r.write_root_state_to_sim(torch.cat([s["pose"], s["vel"]], dim=-1))
            r.write_joint_state_to_sim(s["q"], s["qd"])
            self.env.sim.forward()
            # pretend the last actions were "hold this pose", so the action history and
            # the smoothness terms start from where the robot actually is
            am = self.env.action_manager
            term = am.get_term("joint_pos")
            hold = (s["q"][:, term._target_ids] - term._offset) / term._scale
            for buf in ("_action", "_prev_action", "_prev_prev_action"):
                if hasattr(am, buf):
                    getattr(am, buf)[:] = hold
            term._raw_actions[:] = hold
            self.env.observation_manager.reset(torch.tensor([0]))

    def qpos(self):
        d = self.robot.data
        q = np.zeros(29)
        q[:7] = d.root_link_pose_w[0].cpu().numpy()
        q[7:] = d.joint_pos[0].cpu().numpy()[self.to_joints]
        return q

    def step(self):
        import torch

        with torch.inference_mode():
            self.wrapped.step(self.policy(self.wrapped.get_observations()))


class Performer:
    """The dancer and the walker: the same robot in two simulators, one active at a time."""

    def __init__(self, dance_policy, reference_dir, walk_policy=WALK_POLICY):
        from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
        from mjrl.replay import script_commands

        import env
        import train

        _cpu_backend()
        self.walker = Robot(tasks.load_env_cfg("jumper.posture", play=True),
                            tasks.load_agent_cfg("jumper.posture"),
                            tasks.load_runner_cls("jumper.posture"), walk_policy)
        self.dancer = Robot(env.env_cfg(reference_dir, play=True), train.ppo_cfg(0),
                            MotionTrackingOnPolicyRunner, dance_policy)
        self.command = {"twist": [0.0, 0.0, 0.0], "posture": [0.0, 0.0, 0.0, 0.10647]}
        script_commands(self.walker.env, lambda t: self.command)

    def seek(self, frame):
        """Jump the dancer's reference to this frame."""
        self.dancer.env.command_manager.get_term("motion").time_steps[:] = int(frame)


# ---------------------------------------------------------------------------
# Reference and performance
# ---------------------------------------------------------------------------

def compose_reference(show, out):
    """Every dance block of the show, back to back, as one clip the policy tracks.
    Returns the trajectory and the first frame of each block."""
    limits = load_limits()
    parts, starts, n = [], {}, 0
    for i, sec in enumerate(show.sections):
        if not isinstance(sec, DanceBlock):
            continue
        tr = Composer(sec.segments, show.tempo, limits, start_beat=sec.start).run(
            t_start=float(show.tempo.time(sec.start)), t_end=float(show.tempo.time(sec.start + sec.beats)))
        if parts:     # half a second of standing between blocks keeps the clip continuous
            pad = Composer([HoldHome(1)], show.tempo, limits).run(t_start=0.0, t_end=0.5)
            parts.append(pad)
            n += len(pad.qpos)
        starts[i] = n
        parts.append(tr)
        n += len(tr.qpos)
    if not parts:
        return None, {}
    qpos = np.concatenate([p.qpos for p in parts])
    traj = Trajectory(np.arange(len(qpos)) * DT, qpos, np.concatenate([p.beats for p in parts]),
                      sum((p.labels for p in parts), []), np.concatenate([p.ik_err for p in parts]),
                      DT, show.tempo.beat_times)
    media = Path(out) / "media"
    media.mkdir(parents=True, exist_ok=True)
    for f in media.glob("*.npz"):
        f.unlink()
    write_clip(traj, media / "show.npz", bpm=125.0, source="dance.py")
    validate_clip(media)
    return traj, starts


def perform(show, starts, policy, media):
    """Run the show with physics. Returns qpos per frame (50 Hz), labels and a log."""
    p = Performer(policy, media)
    times = np.arange(float(show.tempo.time(show.start_beat)), float(show.tempo.time(show.end_beat)), DT)
    qs, labels, log = [], [], {"falls": [], "handoffs": []}
    active, current = None, None
    for t in times:
        b = float(show.tempo.beat(t))
        idx = show.section_at(b)
        sec = show.sections[idx]
        if idx != current:
            if isinstance(sec, DanceBlock):
                if active is p.walker:
                    p.dancer.set_state(p.walker.state())
                p.seek(starts[idx] + int(round((t - float(show.tempo.time(sec.start))) / DT)))
                active = p.dancer
            else:
                if active is p.dancer:
                    p.walker.set_state(p.dancer.state())
                active = p.walker
            log["handoffs"].append({"t": round(float(t), 3), "to": type(sec).__name__})
            current = idx
        if isinstance(sec, WalkSection):
            p.command = walk_command(sec.move, b, float(show.tempo.bpm_at(t)), sec.intensity, sec.period)
            for _ in range(int(round(DT / active.step_dt))):    # the walker runs at 200 Hz
                active.step()
            labels.append(f"walk:{sec.move}")
        else:
            active.step()
            labels.append("dance")
        q = active.qpos()
        up_z = 1.0 - 2.0 * (q[4] ** 2 + q[5] ** 2)
        if (q[2] < 0.05 or up_z < 0.6) and (not log["falls"] or t - log["falls"][-1] > 1.0):
            log["falls"].append(round(float(t), 2))
        qs.append(q)
    return np.array(qs), labels, log


def kinematic(show, ref, starts):
    """The reference laid out on the show's clock (walk sections just stand)."""
    times = np.arange(float(show.tempo.time(show.start_beat)), float(show.tempo.time(show.end_beat)), DT)
    qs, labels = [], []
    for t in times:
        idx = show.section_at(float(show.tempo.beat(t)))
        sec = show.sections[idx]
        if isinstance(sec, DanceBlock):
            f = min(starts[idx] + int(round((t - float(show.tempo.time(sec.start))) / DT)), len(ref.qpos) - 1)
            qs.append(ref.qpos[f])
            labels.append(ref.labels[f])
        else:
            qs.append(home_qpos())
            labels.append(f"walk:{sec.move} (physics only)")
    return np.array(qs), labels


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

class Camera:
    def __init__(self, width=960, height=540, azimuth=200.0, elevation=-14.0, distance=1.1, smooth=0.0):
        self.model = load_model(scene=True)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth, self.cam.elevation, self.cam.distance = azimuth, elevation, distance
        self.smooth = smooth       # 0 follows the robot exactly, 0.97 trails behind it
        self._base = self.model.body("base_link").id
        self._look = None

    def frame(self, qpos, follow=True, label=None):
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        p = self.data.xpos[self._base].copy() if follow else np.zeros(3)
        self._look = p if self._look is None or self.smooth <= 0 else \
            self.smooth * self._look + (1.0 - self.smooth) * p
        self.cam.lookat[:] = [self._look[0], self._look[1], 0.11]
        self.renderer.update_scene(self.data, self.cam)
        img = self.renderer.render().copy()
        return caption(img, label) if label else img


def caption(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, img.shape[0] // 22))
    except OSError:
        font = ImageFont.load_default()
    d.rectangle([0, 0, img.shape[1], img.shape[0] // 14 + 6], fill=(0, 0, 0))
    d.text((6, 3), text, fill=(255, 230, 120), font=font)
    return np.asarray(im)


def write_video(frames, path, fps=30.0, audio=None, audio_offset_s=0.0, audio_tempo=1.0):
    """H.264 mp4; with `audio`, the song from audio_offset_s in, sped up by audio_tempo."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    silent = path.with_suffix(".silent.mp4") if audio else path
    with imageio.get_writer(silent, fps=fps, codec="libx264", quality=8, macro_block_size=8,
                            ffmpeg_log_level="error") as w:
        for f in frames:
            w.append_data(f)
    if audio:
        af = f"atempo={audio_tempo:.6f}" if abs(audio_tempo - 1.0) > 1e-6 else "anull"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-ss", f"{audio_offset_s:.3f}",
                        "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-af", af,
                        "-c:v", "copy", "-c:a", "aac", "-shortest", str(path)], check=True)
        silent.unlink()
    return path


def render(qs, labels, beats, path, music, audio_offset, tempo_scale, follow=True, fps=30.0):
    cam = Camera(smooth=0.97)
    frames = []
    for k in range(int(len(qs) * DT * fps)):
        i = min(int(round(k / fps / DT)), len(qs) - 1)
        b = beats[i]
        flash = "  *" if (b - np.floor(b)) < 0.12 else ""
        frames.append(cam.frame(qs[i], follow=follow,
                                label=f"{labels[i]}   beat {int(np.floor(b)) % 4 + 1}{flash}"))
    return write_video(frames, path, fps=fps, audio=music, audio_offset_s=audio_offset, audio_tempo=tempo_scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("choreo", type=Path)
    ap.add_argument("--tempo", type=float, default=1.0, help="play the song this many times faster")
    ap.add_argument("--policy", type=Path, default=DANCE_POLICY)
    ap.add_argument("--reference", action="store_true", help="render the planned motion, no physics")
    a = ap.parse_args()

    show = load_show(a.choreo)
    policy = a.policy
    if a.tempo != 1.0:
        show = show.scaled(a.tempo)
    out = ROOT / "out" / (a.choreo.stem + ("" if a.tempo == 1.0 else f"_x{a.tempo:g}"))
    out.mkdir(parents=True, exist_ok=True)
    music = show.music if show.music and show.music.is_file() else None
    if show.music and not music:
        print(f"no {show.music.name} here, the video will be silent (see README)")
    t_start = float(show.tempo.time(show.start_beat))
    audio_offset = t_start * a.tempo          # in the original, unstretched song
    t0 = time.time()

    ref, starts = compose_reference(show, out)
    print(f"reference composed [{time.time() - t0:.0f} s]", flush=True)
    kq, kl = kinematic(show, ref, starts)
    beats = np.array([float(show.tempo.beat(t)) for t in t_start + np.arange(len(kq)) * DT])
    follow = any(isinstance(s, WalkSection) for s in show.sections)   # fixed camera if it never walks
    if a.reference:
        path = render(kq, kl, beats, out / "reference.mp4", music, audio_offset, a.tempo, follow)
        print(f"wrote {path} [{time.time() - t0:.0f} s]")
        return

    qs, labels, log = perform(show, starts, policy, out / "media")
    np.save(out / "qpos.npy", qs)
    log.update(policy=str(policy), tempo=a.tempo, seconds=float(len(qs) * DT))
    (out / "log.json").write_text(json.dumps(log, indent=1))
    path = render(qs, labels, beats, out / "show.mp4", music, audio_offset, a.tempo, follow)
    print(f"wrote {path}; falls at {log['falls'] or 'none'} [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
