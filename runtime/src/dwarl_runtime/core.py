"""Observation building, path tracking, action mapping and ONNX inference (see package docstring).

Keep in sync with ``ros/src/pkgs/dwarl_controller/src/policy_core.cpp``; both are checked against the
``golden.json`` episode written during policy preparation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEPLOY_VERSION = 5


# ------------------------------------------------------------------------------------------------ contract
@dataclass
class DeploySpec:
    dir: Path
    onnx_file: Path
    obs_input: str = "obs"
    lidar_input: str = "lidar"
    action_output: str = "actions"
    obs_dim: int = 7
    dt: float = 0.1
    num_rays: int = 180
    angle_min: float = -math.pi
    angle_increment: float = 2 * math.pi / 180
    full_circle: bool = True
    origin: tuple[float, float] = (0.0, 0.0)
    min_range: float = 0.12
    max_range: float = 8.0
    lidar_normalize: str = "linear"
    lidar_history: list[int] = field(default_factory=lambda: [0])
    max_goal_distance: float = 10.0
    v_scale: float = 0.8
    w_scale: float = 1.5
    goal_mode: str = "lookahead"
    lookahead_distance: float = 2.0
    lookahead_distances: list[float] = field(default_factory=lambda: [2.0])
    goal_tolerance: float = 0.3
    window: tuple[int, int] = (-5, 30)
    v_min: float = -0.2
    v_max: float = 0.8
    w_max: float = 1.5
    a_max: float = 1.0
    alpha_max: float = 3.0
    action_mode: str = "absolute"
    robot_id: str = ""
    ros: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    command_feedback: bool = False

    @classmethod
    def load(cls, bundle: str | Path) -> DeploySpec:
        d = Path(bundle)
        f = d / "deploy.yaml"
        if not f.is_file():
            raise FileNotFoundError(f"deploy.yaml not found in {d}")
        y = yaml.safe_load(f.read_text())
        version = y.get("version")
        if version not in (2, 3, DEPLOY_VERSION):
            raise ValueError(f"{f}: unsupported deploy.yaml version {version}")
        if y["robot"]["drive_type"] != "diff":
            raise ValueError("only diff-drive policies are supported")
        m, lc, st, g, a = y["model"], y["lidar"], y["state"], y["goal"], y["action"]
        if version == 5 and (g.get("tracker") != "segment_window_v1" or st.get("command_feedback") is not True):
            raise ValueError("v5 requires segment_window_v1 guidance and command feedback")
        if version < 5 and st.get("command_feedback", False):
            raise ValueError("command feedback requires deploy version 5")
        ins = {i["group"]: i for i in m["inputs"]}
        lidar_normalize = str(lc.get("normalize", "linear")) if version >= 3 else "linear"
        lidar_history = [int(k) for k in lc.get("history", [0])] if version >= 3 else [0]
        lookahead_distances = (
            [float(d) for d in g.get("lookahead_distances", [g["lookahead_distance"]])]
            if version >= 3 else [float(g["lookahead_distance"])]
        )
        action_mode = str(a.get("mode", "absolute")) if version >= 3 else "absolute"
        s = cls(
            dir=d, onnx_file=d / m["onnx"], obs_input=ins["state"]["name"], lidar_input=ins["lidar"]["name"],
            action_output=m["outputs"][0]["name"], obs_dim=int(ins["state"]["shape"][1]), dt=float(y["control"]["dt"]),
            num_rays=int(lc["num_rays"]), angle_min=float(lc["angle_min"]),
            angle_increment=float(lc["angle_increment"]), full_circle=bool(lc["full_circle"]),
            origin=(float(lc["origin"][0]), float(lc["origin"][1])), min_range=float(lc["min_range"]),
            max_range=float(lc["max_range"]), lidar_normalize=lidar_normalize, lidar_history=lidar_history,
            max_goal_distance=float(st["max_goal_distance"]),
            v_scale=float(st["v_scale"]), w_scale=float(st["w_scale"]), goal_mode=str(g["mode"]),
            lookahead_distance=float(g["lookahead_distance"]),
            lookahead_distances=lookahead_distances,
            goal_tolerance=float(g["tolerance"]),
            window=(int(g["projection_window"][0]), int(g["projection_window"][1])), v_min=float(a["v"]["min"]),
            v_max=float(a["v"]["max"]), w_max=float(a["w"]["max"]), a_max=float(a["rate_limit"]["a_max"]),
            alpha_max=float(a["rate_limit"]["alpha_max"]), action_mode=action_mode,
            robot_id=str(y["robot"].get("id", "")),
            ros=dict(y.get("ros") or {}), raw=y,
            command_feedback=version == 5,
        )  # fmt: skip
        if s.lidar_normalize not in ("linear", "log"):
            raise ValueError(f"unknown lidar normalization {s.lidar_normalize}")
        if not s.lidar_history or s.lidar_history[0] != 0 or s.lidar_history != sorted(s.lidar_history):
            raise ValueError("lidar history must be ascending and start with 0")
        if s.action_mode not in ("absolute", "window"):
            raise ValueError(f"unknown action mode {s.action_mode}")
        if s.obs_dim != 3 * len(s.lookahead_distances) + 4 + 2 * s.command_feedback:
            raise ValueError("state observation dimension does not match lookahead distances")
        return s


# ------------------------------------------------------------------------------------------------ lidar
@dataclass
class Scan:
    """A planar range scan (sensor_msgs/LaserScan semantics, REP 117 for special values)."""

    ranges: np.ndarray
    angle_min: float
    angle_increment: float
    range_min: float = 0.0
    range_max: float = 0.0


def laser_matrix(pose: Any = None) -> np.ndarray:
    """4x4 pose of the scan frame in the base frame from a 4x4 matrix, ``(xyz, quat_xyzw)`` or ``(x, y, yaw)``."""
    if pose is None:
        return np.eye(4)
    if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
        return pose.astype(np.float64)
    if len(pose) == 3 and np.ndim(pose[0]) == 0:  # planar (x, y, yaw)
        x, y, yaw = (float(v) for v in pose)
        T = np.eye(4)
        T[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
        T[:2, 3] = x, y
        return T
    xyz, q = pose
    qx, qy, qz, qw = (float(v) for v in q)
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                 [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                 [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]]  # fmt: skip
    T[:3, 3] = [float(v) for v in xyz]
    return T


def _wrap(a):
    return np.remainder(a + np.pi, 2 * np.pi) - np.pi


def lidar_observation(spec: DeploySpec, scan: Scan, laser: Any = None) -> np.ndarray:
    """Resample a scan onto the training rays (bearing from the lidar origin in the base frame, closest return
    per bin, gaps filled from the nearest covered bin) and normalise to [0, 1]."""
    T = laser_matrix(laser)
    nr, inc = spec.num_rays, spec.angle_increment
    r = np.asarray(scan.ranges, dtype=np.float64)
    th = scan.angle_min + np.arange(len(r)) * scan.angle_increment
    valid = np.isfinite(r) & (r >= scan.range_min) & ((scan.range_max <= 0) | (r <= scan.range_max))
    rr = np.where(np.isneginf(r), scan.range_min, np.where(valid, r, np.inf))
    hit = np.isfinite(rr)
    d = np.where(hit, rr, max(spec.max_range, 1.0) * 10.0)
    u = T[:3, :3] @ np.stack([np.cos(th), np.sin(th), np.zeros_like(th)])  # (3, n) beam directions in base
    px = T[0, 3] + d * u[0] - spec.origin[0]
    py = T[1, 3] + d * u[1] - spec.origin[1]
    rel = _wrap(np.arctan2(py, px) - spec.angle_min - np.pi) + np.pi  # [0, 2pi)
    k = np.floor(rel / inc + 0.5).astype(np.int64)  # == std::lround for rel >= 0
    if spec.full_circle:
        k %= nr
        keep = np.ones_like(k, dtype=bool)
    else:
        keep = k < nr
    best = np.full(nr, np.inf)
    seen = np.zeros(nr, dtype=bool)
    seen[k[keep]] = True
    np.minimum.at(best, k[keep & hit], np.hypot(px, py)[keep & hit])
    reach = max(1, math.ceil(abs(scan.angle_increment) / inc))
    filled = best.copy()
    for i in np.flatnonzero(~seen):
        for o in range(1, reach + 1):
            lo, hi = i - o, i + o
            if spec.full_circle:
                lo, hi = lo % nr, hi % nr
            lok, hik = 0 <= lo < nr and seen[lo], 0 <= hi < nr and seen[hi]
            if lok or hik:
                filled[i] = min(best[lo] if lok else np.inf, best[hi] if hik else np.inf)
                break
    out = np.clip(np.where(np.isfinite(filled), filled, spec.max_range), spec.min_range, spec.max_range)
    if spec.lidar_normalize == "linear":
        out = (out - spec.min_range) / (spec.max_range - spec.min_range)
    else:
        out = np.log1p(np.maximum(out - spec.min_range, 0.0) / 0.1) / np.log1p(
            (spec.max_range - spec.min_range) / 0.1)
    return out.astype(np.float32)


# ------------------------------------------------------------------------------------------------ path
class PathTracker:
    """Projects the robot onto the plan like training (windowed search around the last segment)."""

    def __init__(self) -> None:
        self.pts = np.zeros((0, 2))
        self.s = np.zeros(0)
        self.seg, self.fresh = 0, True

    def set_plan(self, plan) -> None:
        self.pts = np.asarray(plan, dtype=np.float64).reshape(-1, 2)
        if not len(self.pts) or not np.isfinite(self.pts).all():
            raise ValueError("plan must contain finite points")
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)]) if len(self.pts) else np.zeros(0)
        self.seg, self.fresh = 0, True

    @property
    def empty(self) -> bool:
        return len(self.pts) == 0

    def project(self, xy, window: tuple[int, int]) -> float:
        if not np.isfinite(xy).all():
            raise ValueError("nonfinite robot position")
        if len(self.pts) < 2:
            return 0.0
        last = len(self.pts) - 2
        lo, hi = (0, last + 1) if self.fresh else (self.seg + window[0], self.seg + window[1])
        self.fresh = False
        c = np.clip(np.arange(lo, hi), 0, last)
        a, b = self.pts[c], self.pts[c + 1]
        ab = b - a
        l2 = np.maximum((ab**2).sum(-1), 1e-12)
        t = np.clip(((np.asarray(xy) - a) * ab).sum(-1) / l2, 0.0, 1.0)
        d = np.hypot(*(np.asarray(xy) - (a + t[:, None] * ab)).T)
        j = int(np.argmin(d))  # first minimum, like the C++ strict '<'
        self.seg = int(c[j])
        return float(self.s[c[j]] + t[j] * math.sqrt(l2[j]))

    def point_at(self, s: float) -> np.ndarray:
        return np.array([np.interp(s, self.s, self.pts[:, 0]), np.interp(s, self.s, self.pts[:, 1])])

    @property
    def goal(self) -> np.ndarray:
        return self.pts[-1]


def state_observation(spec: DeploySpec, tracker: PathTracker, pose, v: float, w: float,
                      last_action, command=None) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Goal features, last action, measured velocity and optional previous rate-limited command."""
    x, y, yaw = (float(p) for p in pose)
    s = tracker.project((x, y), spec.window)
    goal = tracker.point_at(s + spec.lookahead_distance) if spec.goal_mode == "lookahead" else tracker.goal
    goals = [tracker.point_at(s + d) for d in spec.lookahead_distances] if spec.goal_mode == "lookahead" else [goal]
    c, sn = math.cos(yaw), math.sin(yaw)
    gm = spec.max_goal_distance
    st = []
    for point in goals:
        dx, dy = point[0] - x, point[1] - y
        gx, gy = c * dx + sn * dy, -sn * dx + c * dy
        dist, ang = math.hypot(gx, gy), math.atan2(gy, gx)
        st.extend([min(dist, gm) / gm, math.sin(ang), math.cos(ang)])
    st.extend([last_action[0], last_action[1], v / spec.v_scale, w / spec.w_scale])
    if spec.command_feedback:
        if command is None:
            raise ValueError("v5 observation requires previous rate-limited command")
        st.extend([command[0] / spec.v_scale, command[1] / spec.w_scale])
    return np.asarray(st, dtype=np.float32), goal


def command_from_action(spec: DeploySpec, action, cmd) -> np.ndarray:
    """Action in [-1, 1]^2 -> rate-limited (v, w) command, exactly as the training env."""
    a0, a1 = (float(np.clip(a, -1.0, 1.0)) for a in action)
    lv, lw = spec.a_max * spec.dt, spec.alpha_max * spec.dt
    if spec.action_mode == "window":
        return np.array([np.clip(cmd[0] + a0 * lv, spec.v_min, spec.v_max),
                         np.clip(cmd[1] + a1 * lw, -spec.w_max, spec.w_max)])
    tv = spec.v_min + (a0 + 1.0) * 0.5 * (spec.v_max - spec.v_min)
    tw = a1 * spec.w_max
    return np.array([cmd[0] + np.clip(tv - cmd[0], -lv, lv), cmd[1] + np.clip(tw - cmd[1], -lw, lw)])


# ------------------------------------------------------------------------------------------------ policy
@dataclass
class Command:
    v: float
    w: float
    action: np.ndarray
    goal: np.ndarray  # lookahead point (plan frame)
    goal_reached: bool


class Policy:
    """Stateful controller: call :meth:`set_plan` for a new path and :meth:`step` every ``dt`` seconds."""

    def __init__(self, bundle: str | Path, threads: int = 1, expected_robot_id: str | None = None) -> None:
        import onnxruntime as ort

        self.spec = DeploySpec.load(bundle)
        if expected_robot_id and self.spec.robot_id != expected_robot_id:
            raise ValueError(f"bundle {bundle} was trained for robot '{self.spec.robot_id}', "
                             f"not '{expected_robot_id}'")  # fmt: skip
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = threads
        opt.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.spec.onnx_file), opt, providers=["CPUExecutionProvider"])
        self.tracker = PathTracker()
        self.reset()

    @property
    def dt(self) -> float:
        return self.spec.dt

    def reset(self) -> None:
        """Start a new episode (training resets the last action and command)."""
        self.last_action = np.zeros(2)
        self.cmd = np.zeros(2)
        self._scan_buf: np.ndarray | None = None

    def set_plan(self, plan, new_goal: bool | None = None) -> None:
        plan = np.asarray(plan, dtype=np.float64).reshape(-1, 2)
        if len(plan) == 0:
            raise ValueError("empty plan")
        if not np.isfinite(plan).all():
            raise ValueError("nonfinite plan")
        if new_goal is None:
            new_goal = self.tracker.empty or np.hypot(*(plan[-1] - self.tracker.goal)) > self.spec.goal_tolerance
        if new_goal:
            self.reset()
        self.tracker.set_plan(plan)

    def act(self, state: np.ndarray, lidar: np.ndarray) -> np.ndarray:
        s = self.spec
        if not np.isfinite(state).all() or not np.isfinite(lidar).all():
            self.cmd[:] = 0.
            raise ValueError("nonfinite policy observation")
        out = self.session.run(
            [s.action_output],
            {
                s.obs_input: state.reshape(1, -1).astype(np.float32),
                s.lidar_input: lidar.reshape(1, 1, len(s.lidar_history), s.num_rays).astype(np.float32),
            },
        )
        action = np.asarray(out[0][0], dtype=np.float64)
        if not np.isfinite(action).all():
            self.cmd[:] = 0.
            raise ValueError("nonfinite policy output")
        return np.clip(action, -1.0, 1.0)

    def step(self, scan: Scan, laser_pose, robot_pose, velocity) -> Command:
        """One control step. ``robot_pose`` = (x, y, yaw) in the plan frame, ``velocity`` = measured (v, w)."""
        if self.tracker.empty:
            raise RuntimeError("no plan: call set_plan() first")
        lidar = lidar_observation(self.spec, scan, laser_pose)
        if self._scan_buf is None:
            self._scan_buf = np.repeat(lidar[None], self.spec.lidar_history[-1] + 1, axis=0)
        else:
            self._scan_buf = np.concatenate([lidar[None], self._scan_buf[:-1]], axis=0)
        lidar = self._scan_buf[self.spec.lidar_history]
        state, goal = state_observation(self.spec, self.tracker, robot_pose, velocity[0], velocity[1],
                                        self.last_action, self.cmd)  # fmt: skip
        a = self.act(state, lidar)
        self.cmd = command_from_action(self.spec, a, self.cmd)
        self.last_action = a
        reached = math.hypot(robot_pose[0] - self.tracker.goal[0], robot_pose[1] - self.tracker.goal[1])
        return Command(float(self.cmd[0]), float(self.cmd[1]), a, goal, reached <= self.spec.goal_tolerance)
