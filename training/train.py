"""Train the dance policy with PPO (rsl_rl), 4096 robots at once on one GPU.

The policy learns to follow the reference in clip/ (training/make_clip.py). We don't start from
scratch but from the dance policy that ships with the jumper repo: it already knows how
to stand and wave its claws on this body, so it gets to "dancing to the clip" much
faster. After adding moves to the library, start from our policy instead and train
less (--init checkpoint/policy.pt --iterations 5000).

    python training/train.py                                # ~4 h on an RTX 4090
    python training/train.py --iterations 200 --envs 512    # smoke test
    python training/train.py --resume logs/<run>/model_5000.pt

Checkpoints go to logs/<date-time>/model_<iteration>.pt every 50 iterations, and
TensorBoard logs next to them (tensorboard --logdir logs).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))           # so `training` imports from here

import tasks  # noqa: E402
from mjrl.backend import resolve  # noqa: E402
from mjrl.backend.select import use_backend  # noqa: E402
JUMPER_DANCE = Path(tasks.__file__).parent / "jumper/dance/out/example/model_2300.pt"


def ppo_cfg(iterations):
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

    net = dict(hidden_dims=(512, 256, 128, 64), activation="elu", obs_normalization=True)
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(**net, distribution_cfg={"class_name": "GaussianDistribution",
                                                     "init_std": 1.0, "std_type": "scalar"}),
        critic=RslRlModelCfg(**net),
        algorithm=RslRlPpoAlgorithmCfg(
            learning_rate=1.0e-3, schedule="adaptive", desired_kl=0.01,
            gamma=0.99, lam=0.95, clip_param=0.2,
            value_loss_coef=1.0, use_clipped_value_loss=True,
            # half the usual 0.01: the reference already says what to do, so it needs
            # less exploring, and too much noise makes the dance mushy
            entropy_coef=0.005,
            num_learning_epochs=5, num_mini_batches=4, max_grad_norm=1.0,
            # no left/right mirroring: a dance isn't symmetric, raising the left claw
            # is not as good as raising the right one
            symmetry_cfg=None,
        ),
        experiment_name="dance", num_steps_per_env=24, save_interval=50,
        max_iterations=iterations, logger="tensorboard",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=20_000)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--device", default="auto", help="cuda:0, cpu, ...")
    ap.add_argument("--init", type=Path, default=JUMPER_DANCE, help="checkpoint to start from")
    ap.add_argument("--resume", type=Path, help="continue one of our own runs instead")
    a = ap.parse_args()

    res = resolve(device=a.device, num_envs=a.envs, training=True)
    use_backend(res)       # MuJoCo Warp on a GPU, plain MuJoCo threads on a CPU

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

    from training.env import env_cfg

    cfg = env_cfg(ROOT / "clip")
    cfg.scene.num_envs = res.num_envs
    log_dir = ROOT / "logs" / f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    log_dir.mkdir(parents=True)
    print(f"logging to {log_dir}")

    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)
    try:
        runner = MotionTrackingOnPolicyRunner(RslRlVecEnvWrapper(env), asdict(ppo_cfg(a.iterations)),
                                              str(log_dir), res.device)
        start = a.resume or a.init
        if start:
            runner.load(str(start), map_location=res.device)
            print(f"starting from {start} (iteration {runner.current_learning_iteration})")
        runner.learn(num_learning_iterations=a.iterations, init_at_random_ep_len=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
