"""Read and validate the deployment contract before a bundle is cached or installed."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEPLOY_VERSION = 5


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
