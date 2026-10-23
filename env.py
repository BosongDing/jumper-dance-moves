"""The RL environment: Jumper tracking a dance clip, in MuJoCo (via mjlab).

This is motion tracking in the BeyondMimic style. The policy is shown the reference
dance (where every joint should be now and a bit later) and its own sensors, and is
paid for matching it while staying upright, smooth and gentle on the motors.

What the policy sees (209 numbers, 50 times a second):
  the reference's joint angles now, its joint speeds, and its joint angles
  40 / 100 / 200 ms ahead; how far the body is tilted compared to the reference;
  gyro and gravity direction from the IMU; its own joint angles and speeds; its last
  action; the motor torques.

What it's paid for:
  + body position / orientation / velocity close to the reference   (6 terms)
  + joint angles and joint speeds close to the reference           (2 terms)
  - jerky actions, electrical power, torque above the servo rating,
    hitting joint limits, legs bumping into each other

When an episode ends early:
  body height off by more than the clip's whole height range, tilted 45 deg away from
  the reference, or a standing foot 5 cm off the ground. Otherwise after 10 s.

Randomized: foot friction, centre of mass, joint encoder offsets, a gentle push every
3-8 s, and small errors in the starting pose.

Most numbers below are scaled to the robot. It's 0.1 m tall, and mjlab's defaults
were made for a 1.3 m humanoid.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg as Obs
from mjlab.managers.reward_manager import RewardTermCfg as Rew
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Noise
from tasks.jumper.common.constants import (ACTION_SCALE, ACTUATOR_FORCE_NOISE, FEET, FOOT_GEOMS, HOME,
                                           IMU_ANG_VEL_NOISE, IMU_GRAVITY_NOISE,
                                           JUMPER_FOOT_FRICTION_RANGE, get_jumper_robot_cfg)
from tasks.jumper.common.dance import observations as ref
from tasks.jumper.common.dance import rewards as costs
from tasks.jumper.common.dance.motion import ensure_motion_npz
from tasks.jumper.common.mdp.observations import actuator_force

SUPPORT_FEET = ("LM_foot_tip_link", "RM_foot_tip_link", "LR_foot_tip_link", "RR_foot_tip_link")
EPISODE_S = 10.0


def constant_phase(env, command_name):
    # Jumper's own dance task tells the policy where it is in its one song. Ours has to
    # dance songs it never trained on, so this is always (0, 1). It stays in the
    # observation only so the layout matches jumper's dance policy, which we start from.
    out = torch.zeros(env.num_envs, 2, device=env.device)
    out[:, 1] = 1.0
    return out


def clip_stats(clip_dir):
    """How much the clip moves, measured by make_clip.py: the joints' typical swing (rad)
    and speed (rad/s), and the lowest and highest the body goes (m). The tracking widths
    and the height limit below are set from these."""
    path = Path(clip_dir) / "stats.json"
    if path.is_file():
        st = json.loads(path.read_text())
        return st["joint_pos_rms"], st["joint_vel_rms"], tuple(st["root_z"])
    # Only playing back (dance.py): rewards and the height limit aren't used, any values do.
    return 0.4, 2.3, (0.08, 0.16)


def env_cfg(clip_dir, play=False):
    joint_pos_rms, joint_vel_rms, root_z = clip_stats(clip_dir)
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": get_jumper_robot_cfg(asset=None)}
    cfg.sim.njmax = 512
    cfg.sim.nconmax = 128
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            fields=("found", "force"), reduce="none", num_slots=1, history_length=cfg.decimation),
    )

    # All 22 joints, claws and pincers included: target = HOME + 0.25 * action
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.actuator_names = list(HOME)
    action.scale = {j: ACTION_SCALE for j in HOME}

    # The clip, converted by jumper's loader (it also checks limits and continuity)
    control_dt = cfg.decimation * cfg.sim.mujoco.timestep
    motion = cfg.commands["motion"]
    motion.motion_file = str(ensure_motion_npz(control_dt, Path(clip_dir)))
    motion.anchor_body_name = "base_link"
    motion.body_names = ("base_link",) + FEET
    # every reset starts near a random frame of the clip, nudged by up to:
    motion.pose_range = {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005),
                         "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05)}
    motion.velocity_range = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
                             "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2)}
    motion.joint_position_range = (-0.05, 0.05)

    # --- observations -------------------------------------------------------------
    c = {"command_name": "motion"}
    reference = {
        "clip_phase": Obs(func=constant_phase, params=c),
        "ref_joint_pos": Obs(func=ref.ref_joint_pos, params=c),
        "ref_joint_vel": Obs(func=ref.ref_joint_vel, params=c),
        "ref_future": Obs(func=ref.ref_future, params={**c, "horizons": ref.FUTURE_HORIZONS}),
        "ref_tilt_error": Obs(func=ref.ref_tilt_error, params=c),
    }

    def own_sensors(noisy):
        def n(x):
            return Noise(n_min=-x, n_max=x) if noisy else None
        return {
            "base_ang_vel": Obs(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"},
                                noise=n(IMU_ANG_VEL_NOISE)),
            "projected_gravity": Obs(func=mdp.projected_gravity, noise=n(IMU_GRAVITY_NOISE)),
            "joint_pos": Obs(func=mdp.joint_pos_rel, noise=n(0.01)),
            "joint_vel": Obs(func=mdp.joint_vel_rel, noise=n(0.5)),
            "actions": Obs(func=mdp.last_action),
            "actuator_force": Obs(func=actuator_force, noise=n(ACTUATOR_FORCE_NOISE)),
        }

    cfg.observations["actor"] = ObservationGroupCfg(
        terms={**reference, **own_sensors(noisy=True)}, concatenate_terms=True, enable_corruption=True)
    # the critic only exists in training, so it also gets things a real robot can't measure
    cfg.observations["critic"] = ObservationGroupCfg(
        terms={**reference, **own_sensors(noisy=False),
               "base_lin_vel": Obs(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}),
               "body_pos": Obs(func=mdp.robot_body_pos_b, params=c),
               "body_ori": Obs(func=mdp.robot_body_ori_b, params=c)},
        concatenate_terms=True, enable_corruption=False)

    # --- rewards ------------------------------------------------------------------
    # Each tracking term is exp(-(error / std)^2). The std is "how wrong is still
    # mostly fine": BeyondMimic's are humanoid-sized (0.3 m) and on this robot they
    # let it skip a 3 cm sway for free, so the body ones are much tighter here.
    every_motor = {"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))}
    cfg.rewards = {
        "motion_global_root_pos": Rew(func=mdp.motion_global_anchor_position_error_exp,
                                      weight=0.5, params={**c, "std": 0.05}),
        "motion_global_root_ori": Rew(func=mdp.motion_global_anchor_orientation_error_exp,
                                      weight=0.5, params={**c, "std": 0.2}),
        "motion_body_pos": Rew(func=mdp.motion_relative_body_position_error_exp,
                               weight=1.0, params={**c, "std": 0.05}),
        "motion_body_ori": Rew(func=mdp.motion_relative_body_orientation_error_exp,
                               weight=1.0, params={**c, "std": 0.25}),
        "motion_body_lin_vel": Rew(func=mdp.motion_global_body_linear_velocity_error_exp,
                                   weight=1.0, params={**c, "std": 0.5}),
        "motion_body_ang_vel": Rew(func=mdp.motion_global_body_angular_velocity_error_exp,
                                   weight=1.0, params={**c, "std": 3.14}),
        # joint space: std a bit under half the clip's joint amplitude, so standing
        # still scores near zero; speed is looser and half the weight, the pose wins
        "motion_joint_pos": Rew(func=costs.motion_joint_pos_error_exp,
                                weight=1.0, params={**c, "std": round(0.44 * joint_pos_rms, 3)}),
        "motion_joint_vel": Rew(func=costs.motion_joint_vel_error_exp,
                                weight=0.5, params={**c, "std": round(0.66 * joint_vel_rms, 3)}),
        # total positive reward is 6.5 per step; the costs are sized against that
        "action_rate_l2": Rew(func=mdp.action_rate_l2, weight=-0.1),
        "action_acc_l2": Rew(func=mdp.action_acc_l2, weight=-0.05),
        "power": Rew(func=costs.actuator_power_cost, weight=-0.02, params=every_motor),
        # one motor stuck at its 1.75 N*m peak costs about 12% of the positive reward
        "torque_headroom": Rew(func=costs.torque_headroom_cost, weight=-2.0, params=every_motor),
        "joint_limit": Rew(func=mdp.joint_pos_limits, weight=-10.0,
                           params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
        # 1 N, not mjlab's 10 N: the legs can only push each other with ~12 N at most
        "self_collisions": Rew(func=mdp.self_collision_cost, weight=-1.0,
                               params={"sensor_name": "self_collision", "force_threshold": 1.0}),
    }

    # --- when to give up ----------------------------------------------------------
    t = cfg.terminations
    t["anchor_pos"].params["threshold"] = round(root_z[1] - root_z[0] + 0.008, 3)
    t["anchor_ori"].params["threshold"] = 0.3        # 1 - cos(tilt): about 45 degrees
    t["ee_body_pos"].params["body_names"] = SUPPORT_FEET
    t["ee_body_pos"].params["threshold"] = 0.05      # m, half its standing height

    # --- randomization ------------------------------------------------------------
    e = cfg.events
    e["foot_friction"].params["asset_cfg"].geom_names = list(FOOT_GEOMS)
    e["foot_friction"].params["ranges"] = JUMPER_FOOT_FRICTION_RANGE
    e["base_com"].params["asset_cfg"].body_names = ("base_link",)
    # mjlab shoves every 1-3 s at up to 0.5 m/s, which knocks a 2 kg robot over.
    # Gentler and rarer: enough that it can't just replay the clip open-loop.
    e["push_robot"].interval_range_s = (3.0, 8.0)
    e["push_robot"].params["velocity_range"] = {
        "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
        "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2)}

    cfg.metrics["peak_torque"] = MetricsTermCfg(func=costs.peak_actuator_force, reduce="max",
                                                params=every_motor)
    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 0.9
    cfg.episode_length_s = EPISODE_S

    if play:
        # dance the whole clip from the first frame, no noise, no pushes
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        motion.pose_range = {}
        motion.velocity_range = {}
        motion.joint_position_range = (0.0, 0.0)
        motion.sampling_mode = "start"
    return cfg
