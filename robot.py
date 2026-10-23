"""Jumper's body: the MuJoCo model, forward/inverse kinematics and balance checks.

The model comes straight from the jumper repo, so a change there flows through.
qpos layout: [0:3] base position, [3:7] base quaternion (w, x, y, z), [7:29] the 22
joints in HOME order: LF_J0..J4, RF_J0..J4, LM_J0..J2, RM_J0..J2, LR_J0..J2, RR_J0..J2.
LF / RF are the two claw arms, the other four are the legs it stands on while dancing.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import mink
import mujoco
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

warnings.filterwarnings("ignore", message=".*torch.jit.script.*")

from tasks.jumper.common.constants import FOOT_SITE_Z, HOME, STAND_Z, get_spec  # noqa: E402

JOINTS = tuple(HOME)
HOME_Q = np.array([HOME[j] for j in JOINTS])
ARMS = ("LF", "RF")
SUPPORT = ("LM", "RM", "LR", "RR")
LEGS = ARMS + SUPPORT
GRIPPER = {"LF": "LF_J4_joint", "RF": "RF_J4_joint"}
ARM_SLICE = {"LF": slice(0, 5), "RF": slice(5, 10)}   # into JOINTS / qpos[7:]
CHASSIS = ("base_link", "upper_shell_link", "display_module_link", "camera_link", "tof_sensor_link")


def jidx(name):
    return JOINTS.index(name)


def quat_from_rpy(roll, pitch, yaw):
    """wxyz quaternion for R = Rz(yaw) Ry(pitch) Rx(roll)."""
    x, y, z, w = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    return np.array([w, x, y, z])


def rpy_from_quat(q):
    w, x, y, z = q
    yaw, pitch, roll = Rotation.from_quat([x, y, z, w]).as_euler("ZYX")
    return np.array([roll, pitch, yaw])


def quat_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def home_qpos():
    q = np.zeros(29)
    q[2] = STAND_Z
    q[3] = 1.0
    q[7:] = HOME_Q
    return q


@lru_cache(maxsize=None)
def _compiled(scene):
    spec = get_spec(None)
    if scene:
        _add_beach(spec)
    model = spec.compile()
    assert tuple(model.joint(i).name for i in range(1, model.njnt)) == JOINTS
    return model


def load_model(scene=False):
    """A fresh copy of the robot model, optionally standing on sand under a dusk sky."""
    return copy.deepcopy(_compiled(scene))


def _add_beach(spec):
    spec.visual.headlight.ambient = [0.42, 0.40, 0.45]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.quality.shadowsize = 4096
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.30, 0.38, 0.62], rgb2=[0.98, 0.72, 0.45], width=512, height=512)
    spec.add_texture(name="sand", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, mark=mujoco.mjtMark.mjMARK_RANDOM,
                     random=0.08, rgb1=[0.93, 0.83, 0.64], rgb2=[0.93, 0.83, 0.64],
                     markrgb=[0.80, 0.68, 0.50], width=512, height=512)
    mat = spec.add_material(name="sand", texrepeat=[4, 4], texuniform=True, reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "sand"
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, 0.05], material="sand", contype=0, conaffinity=0)
    spec.worldbody.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             dir=[-0.5, 0.4, -0.75], diffuse=[0.95, 0.80, 0.62],
                             specular=[0.2, 0.15, 0.1], castshadow=True)


class Kin:
    """Forward kinematics plus the few geometric questions the moves keep asking."""

    def __init__(self, model=None):
        self.model = m = model if model is not None else load_model()
        self.data = mujoco.MjData(m)
        self.site = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in LEGS}
        self.body = {m.body(i).name: i for i in range(m.nbody)}

        def body_of(g):
            return m.body(m.geom_bodyid[g]).name

        self.chassis_geoms = [g for g in range(m.ngeom) if body_of(g) in CHASSIS and m.geom_contype[g]]
        # forearm and claw only: the shoulder and upper arm sit next to the shell anyway
        self.arm_geoms = {a: [g for g in range(m.ngeom)
                              if body_of(g).startswith(a)
                              and (m.geom_contype[g] or m.geom_conaffinity[g])
                              and not body_of(g).endswith(("shoulder_link", "upper_arm_link"))]
                          for a in ARMS}

    def set(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def site_pos(self, name):
        return self.data.site_xpos[self.site[name]].copy()

    def body_pos(self, name):
        return self.data.xpos[self.body[name]].copy()

    def com(self):
        return self.data.subtree_com[self.body["base_link"]].copy()

    def chassis_clearance(self, arm):
        """Closest distance between an arm and the shell. MuJoCo doesn't collide the
        two, so without this check a claw can pass straight through the body."""
        return min(geom_distance(self.model, self.data, ga, gc, 0.05)
                   for ga in self.arm_geoms[arm] for gc in self.chassis_geoms)

    def claw_gap(self):
        return min(geom_distance(self.model, self.data, ga, gb, 0.1)
                   for ga in self.arm_geoms["LF"] for gb in self.arm_geoms["RF"])


def geom_distance(model, data, ga, gb, dmax):
    # For a few separated mesh pairs (forearm vs shell) mj_geomDistance returns exactly
    # 0 while its own closest points are centimetres apart; fall back to those, then to
    # the nearest pair of vertices.
    fromto = np.zeros(6)
    d = mujoco.mj_geomDistance(model, data, ga, gb, dmax, fromto)
    if d != 0.0:
        return float(d)
    seg = float(np.linalg.norm(fromto[3:] - fromto[:3]))
    if seg > 1e-4:
        return seg
    mesh = mujoco.mjtGeom.mjGEOM_MESH
    if model.geom_type[ga] != mesh or model.geom_type[gb] != mesh:
        return 0.0
    va, vb = _mesh_vertices(model, data, ga), _mesh_vertices(model, data, gb)
    return min(dmax, float(cKDTree(vb).query(va)[0].min()))


def _mesh_vertices(model, data, g):
    mid = model.geom_dataid[g]
    a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    return model.mesh_vert[a:a + n] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]


def polygon_margin(points_xy, p):
    """Signed distance of p inside the convex hull of points_xy (positive = inside)."""
    eq = ConvexHull(points_xy).equations
    return float(-(eq[:, :2] @ p + eq[:, 2]).max())


# ---------------------------------------------------------------------------
# Stances
#
# HOME is how Jumper stands to walk: six feet down, claws on the sand. To dance, the
# four legs step forward under the body first (otherwise the COM is only 1 cm in front
# of the rear feet and lifting the claws tips it over). DANCE is that stance, taken
# from the crab dance that ships with the jumper repo.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stance:
    base_z: float
    pitch: float           # rad, + is nose down
    feet_xy: dict          # leg -> (x, y) with the base at the origin
    leg_q: np.ndarray      # preferred leg joints (LM, RM, LR, RR)
    feet_z: float = FOOT_SITE_Z

    def foot(self, leg):
        x, y = self.feet_xy[leg]
        return np.array([x, y, self.feet_z])


DANCE = Stance(
    base_z=0.135,
    pitch=np.deg2rad(-6.0),
    feet_xy={"LM": (0.056, 0.184), "RM": (0.056, -0.184),
             "LR": (-0.060, 0.183), "RR": (-0.060, -0.183)},
    leg_q=np.array([-0.40, -0.175, -1.21, 0.40, 0.175, 1.21,
                    0.06, 0.076, -1.46, -0.06, -0.076, 1.46]),
)


def _home_feet():
    k = Kin()
    k.set(home_qpos())
    return {s: k.site_pos(s) for s in LEGS}


@dataclass(frozen=True)
class HomeStance:
    base_z: float = STAND_Z
    q: np.ndarray = field(default_factory=lambda: HOME_Q.copy())
    feet: dict = field(default_factory=_home_feet)

    @property
    def leg_q(self):
        return self.q[10:].copy()

    def arm_q(self, arm):
        return self.q[0:5].copy() if arm == "LF" else self.q[5:10].copy()


HOME_STANCE = HomeStance()


def base_pose(off):
    """World base position and quaternion for an offset [dx dy dz roll pitch yaw] from
    the DANCE stance."""
    return (np.array([off[0], off[1], DANCE.base_z + off[2]]),
            quat_from_rpy(off[3], DANCE.pitch + off[4], off[5]))


_tip_kin = None


def arm_tips_base(arm_q):
    """Palm-pad positions in the base frame for the given arm joints."""
    global _tip_kin
    if _tip_kin is None:
        _tip_kin = Kin()
    q = home_qpos()
    q[:3] = 0.0
    q[7:12], q[12:17] = arm_q["LF"], arm_q["RF"]
    _tip_kin.set(q)
    return {a: _tip_kin.site_pos(a) for a in ARMS}


# The right arm is the left arm's mirror image: joints J0, J2, J3, J4 change sign.
MIRROR = np.array([-1.0, 1.0, -1.0, -1.0, -1.0])


def mirror(q_left):
    return q_left * MIRROR


# ---------------------------------------------------------------------------
# Whole-body IK
# ---------------------------------------------------------------------------

@dataclass
class Targets:
    base_pos: np.ndarray                     # (3,) world
    base_quat: np.ndarray                    # (4,) wxyz
    feet: dict                               # support leg -> (3,) world
    arm_q: dict                              # arm -> (5,) joint keyframe J0..J4
    claw_tip: dict = field(default_factory=dict)   # arm -> (3,) world palm-pad target
    leg_q: np.ndarray | None = None          # (12,) loose preference for the legs


@dataclass
class Costs:
    foot: float = 3000.0          # planted feet must not slide
    base_pos: float = 40.0
    base_ori: float = 15.0
    claw_tip: float = 40.0
    arm_posture: float = 8.0       # arms follow their joint keyframe...
    arm_posture_soft: float = 0.2  # ...loosely, when a claw tip has a world target
    leg_posture: float = 0.03


class WholeBodyIK:
    """Per frame: base pose, four planted feet, and each arm either following a joint
    keyframe or reaching for a point (a drum hit on the sand, an overhead clap).

    The arms are authored in joint space on purpose: an arm has several ways to put its
    claw in the same spot, and the keyframe is what picks the one that looks right.
    Hard limits keep the arms off the shell and off each other.
    """

    def __init__(self, leg_q, costs=None, limit_margin=0.02, clearance=0.006, arm_gap=0.004):
        self.model = m = load_model()
        self.kin = Kin(m)
        self.costs = c = costs or Costs()
        self.cfg = mink.Configuration(m)
        self.feet = {s: mink.FrameTask(s, "site", c.foot, 0.0) for s in SUPPORT}
        self.base = mink.FrameTask("base_link", "body", c.base_pos, c.base_ori)
        self.tip = {a: mink.FrameTask(a, "site", c.claw_tip, 0.0) for a in ARMS}
        self.posture = mink.PostureTask(m, cost=1.0)
        self.leg_q = np.asarray(leg_q, dtype=float)
        chassis = [m.geom(g).name for g in self.kin.chassis_geoms]
        arm = {a: [m.geom(g).name for g in self.kin.arm_geoms[a]] for a in ARMS}
        self.limits = [
            mink.ConfigurationLimit(m, gain=0.95, min_distance_from_limits=limit_margin),
            mink.CollisionAvoidanceLimit(m, [(arm["LF"], chassis), (arm["RF"], chassis)],
                                         minimum_distance_from_collisions=clearance,
                                         collision_detection_distance=clearance + 0.08),
            mink.CollisionAvoidanceLimit(m, [(arm["LF"], arm["RF"])],
                                         minimum_distance_from_collisions=arm_gap,
                                         collision_detection_distance=arm_gap + 0.08),
        ]
        self.q = None
        self.max_step = 0.08   # rad per iteration
        # rad per frame: 78% of the servo's top speed (30.7 rad/s) at 50 Hz.
        # None for one-off poses that aren't part of a trajectory.
        self.max_joint_step = 0.78 * 30.735 * 0.02

    def reset(self, q):
        self.q = q.copy()

    def solve(self, t, iters=30, tol=1e-5, collisions=True):
        c = self.costs
        self.cfg.update(self.q)
        self.base.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3(np.asarray(t.base_quat, dtype=float)), np.asarray(t.base_pos, dtype=float)))
        tasks = [self.base, self.posture]
        for s in SUPPORT:
            self.feet[s].set_target(mink.SE3.from_translation(t.feet[s]))
            tasks.append(self.feet[s])

        target = self.q.copy()
        cost = np.full(len(JOINTS), c.leg_posture)
        target[7 + 10:] = self.leg_q if t.leg_q is None else t.leg_q
        for a in ARMS:
            target[7 + ARM_SLICE[a].start:7 + ARM_SLICE[a].stop] = t.arm_q[a]
            soft = a in t.claw_tip
            cost[ARM_SLICE[a]] = c.arm_posture_soft if soft else c.arm_posture
            cost[jidx(GRIPPER[a])] = 0.0
            if soft:
                self.tip[a].set_target(mink.SE3.from_translation(t.claw_tip[a]))
                tasks.append(self.tip[a])
        self.posture.set_target(target)
        self.posture.set_cost(np.concatenate([np.zeros(6), cost]))

        q_frame = self.q[7:].copy()
        limits = self.limits if collisions else self.limits[:1]
        for _ in range(iters):
            v = mink.solve_ik(self.cfg, tasks, 1.0, "daqp", damping=1e-3, limits=limits)
            # Small steps: the collision limits are linearised, and a big step can jump
            # from outside their detection distance straight into the shell.
            n = np.linalg.norm(v[6:], ord=np.inf)
            if n > self.max_step:
                v *= self.max_step / n
            # Keep the whole frame within servo speed inside the loop too, so every
            # step is one the collision limits have actually checked.
            if self.max_joint_step is not None:
                d = self.cfg.q[7:] - q_frame
                vj = v[6:]
                room = np.where(vj > 0, self.max_joint_step - d, self.max_joint_step + d)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(np.abs(vj) > 1e-12, np.clip(room, 0, None) / np.abs(vj), np.inf)
                scale = min(1.0, float(ratio.min()))
                if scale < 1.0:
                    v = v * scale
            self.cfg.integrate_inplace(v, 1.0)
            if np.linalg.norm(v) < tol:
                break
        q = self.cfg.q.copy()
        for a in ARMS:
            q[7 + jidx(GRIPPER[a])] = t.arm_q[a][4]   # pincers are set directly
        if self.max_joint_step is not None:
            dq = np.clip(q[7:] - self.q[7:], -self.max_joint_step, self.max_joint_step)
            q[7:] = self.q[7:] + dq
        self.q = q
        return q, self.errors(q, t)

    def errors(self, q, t):
        k = self.kin
        k.set(q)
        return {
            "foot": max(float(np.linalg.norm(k.site_pos(s) - t.feet[s])) for s in SUPPORT),
            "base": float(np.linalg.norm(k.body_pos("base_link") - t.base_pos)),
            "tip": {a: float(np.linalg.norm(k.site_pos(a) - p)) for a, p in t.claw_tip.items()},
        }


# ---------------------------------------------------------------------------
# Can the robot actually do this? Checked on the kinematic reference, before physics.
# ---------------------------------------------------------------------------

G = 9.81
SPEED_LIMIT = 0.8 * 30.735   # rad/s, 80% of the servo's top speed

# Touching is fine, going through is not.
THRESH = dict(foot_err=0.004, base_err=0.010, tip_err=0.006, com=0.010, zmp=-0.005,
              clearance=0.0, gap=0.0, speed=SPEED_LIMIT)


@dataclass
class Report:
    com: np.ndarray          # COM margin inside the support polygon, m
    zmp: np.ndarray          # same for the zero-moment point (the dynamic version)
    clearance: np.ndarray    # arm to shell, m
    gap: np.ndarray          # claw to claw, m
    speed: np.ndarray        # fastest joint per step, rad/s
    ik_err: np.ndarray       # (N, 3) foot, claw tip, base

    def summary(self, skip=0):
        s = slice(skip, None)
        return dict(
            foot_err=float(self.ik_err[s, 0].max()), tip_err=float(self.ik_err[s, 1].max()),
            base_err=float(self.ik_err[s, 2].max()), com=float(np.nanmin(self.com[s])),
            zmp=float(np.nanmin(self.zmp[s])), clearance=float(np.nanmin(self.clearance[s])),
            gap=float(np.nanmin(self.gap[s])), speed=float(self.speed[max(skip - 1, 0):].max()))

    def violations(self, skip=0, thresh=None):
        th = dict(THRESH, **(thresh or {}))
        s = self.summary(skip)
        bad = [f"{k}={s[k]:.4g}>{th[k]:.4g}" for k in ("foot_err", "base_err", "tip_err", "speed")
               if s[k] > th[k]]
        bad += [f"{k}={s[k]:.4g}<{th[k]:.4g}" for k in ("com", "zmp", "clearance", "gap")
                if s[k] < th[k]]
        return bad


def evaluate(qpos, dt, ik_err=None, mesh_every=1, smooth=5):
    """Balance, clearance and speed for every frame of a kinematic trajectory.

    The support polygon is the feet within 5 mm of the ground plus any claw within 8 mm
    (a drum stroke really does land on it). The ZMP is what limits fast sways: the
    faster the body swings, the further it throws the ZMP out past the COM.
    """
    k = Kin()
    n = len(qpos)
    com = np.zeros((n, 3))
    margin = np.full(n, np.nan)
    clearance = np.full(n, np.nan)
    gap = np.full(n, np.nan)
    polys = []
    for i, q in enumerate(qpos):
        k.set(q)
        com[i] = k.com()
        pts = [k.site_pos(s)[:2] for s in SUPPORT if k.site_pos(s)[2] < FOOT_SITE_Z + 0.005]
        pts += [k.site_pos(a)[:2] for a in ARMS if k.site_pos(a)[2] < FOOT_SITE_Z + 0.008]
        pts = np.array(pts)
        polys.append(pts)
        margin[i] = polygon_margin(pts, com[i, :2]) if len(pts) >= 3 else np.nan
        if i % mesh_every == 0:
            clearance[i] = min(k.chassis_clearance(a) for a in ARMS)
            gap[i] = k.claw_gap()

    c = com.copy()
    if smooth > 1 and n > smooth:
        ker = np.ones(smooth) / smooth
        for j in range(3):
            c[:, j] = np.convolve(np.pad(com[:, j], smooth // 2, mode="edge"), ker, mode="valid")[:n]
    acc = np.gradient(np.gradient(c, dt, axis=0), dt, axis=0)
    gz = G + acc[:, 2:3]
    zmp_xy = c[:, :2] - c[:, 2:3] * acc[:, :2] / np.maximum(gz, 1e-3)
    # near free fall there's no ground reaction to place, so those frames aren't judged
    zmp = np.array([polygon_margin(p, z) if (len(p) >= 3 and g > 2.0) else np.nan
                    for p, z, g in zip(polys, zmp_xy, gz[:, 0])])
    speed = np.abs(np.diff(qpos[:, 7:], axis=0)).max(axis=1) / dt
    return Report(com=margin, zmp=zmp, clearance=clearance, gap=gap, speed=speed,
                  ik_err=np.zeros((n, 3)) if ik_err is None else ik_err)
