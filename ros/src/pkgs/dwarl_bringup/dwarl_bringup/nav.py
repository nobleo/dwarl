"""Helpers shared by the DWARL launch files (no ROS imports: unit-testable and reusable from robot packages).

* :func:`resolve_bundle` - find a policy bundle (``deploy.yaml`` + ``policy.onnx``) like the C++ plugin does.
* :func:`render_nav2_params` - specialise the generic Nav2 parameter file for the robot a bundle was trained for
  (frames, footprint, topics, rates, limits, collision_monitor zones) from its ``deploy.yaml``.
"""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any

import yaml


def resolve_bundle(model_dir: str) -> Path:
    """Launch files require the already selected immutable bundle."""
    p = Path(model_dir).resolve()
    if not model_dir or not all((p / name).is_file() for name in ("deploy.yaml", "policy.onnx")):
        raise RuntimeError(f"provide model_dir:=<complete policy bundle>, got {model_dir!r}")
    return p


def load_deploy(bundle: str | Path) -> dict[str, Any]:
    d = yaml.safe_load((Path(bundle) / "deploy.yaml").read_text())
    if d.get("version") not in (2, 3, 5):
        raise RuntimeError(f"{bundle}: unsupported deploy.yaml version {d.get('version')}")
    return d


def _footprint(deploy: dict) -> list[list[float]]:
    r = deploy["robot"]
    if r.get("footprint"):
        return [[float(x), float(y)] for x, y in r["footprint"]]
    rad = float(r["footprint_radius"])
    return [[rad * math.cos(k * math.pi / 4 + math.pi / 8), rad * math.sin(k * math.pi / 4 + math.pi / 8)]
            for k in range(8)]  # fmt: skip


def _poly(points) -> str:
    return "[" + ", ".join(f"[{x:.3f}, {y:.3f}]" for x, y in points) + "]"


def _box(x0, x1, y0, y1):
    return [[x1, y1], [x1, y0], [x0, y0], [x0, y1]]


def safety_zones(deploy: dict, margin: float = 0.035, latency: float = 0.15, decel: float | None = None) -> dict:
    """Velocity-dependent collision_monitor STOP zones: footprint bbox + margin, stretched in the driving direction
    by the braking distance v^2 / (2 a) + v * latency at the top speed of each band."""
    fp = _footprint(deploy)
    xs, ys = [p[0] for p in fp], [p[1] for p in fp]
    x0, x1, y0, y1 = min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin
    act = deploy["action"]
    a = decel or float(act["rate_limit"]["a_max"])
    vmax, vmin = float(act["v"]["max"]), float(act["v"]["min"])
    brake = lambda v: v * v / (2 * a) + abs(v) * latency  # noqa: E731
    vmid = 0.5 * vmax
    R = max(math.hypot(x, y) for x, y in fp) + margin
    octagon = [[R * math.cos(k * math.pi / 4 + math.pi / 8), R * math.sin(k * math.pi / 4 + math.pi / 8)]
               for k in range(8)]  # fmt: skip
    zones = {
        "forward_fast": (_box(x0, x1 + brake(vmax), y0, y1), vmid, 10.0, -10.0, 10.0),
        "forward_slow": (_box(x0, x1 + brake(vmid), y0, y1), 0.05, vmid, -10.0, 10.0),
        "backward": (_box(x0 - brake(min(vmin, -0.05)), x1, y0, y1), -10.0, -0.05, -10.0, 10.0),
        "turn_left": (octagon, -0.05, 0.05, 0.2, 10.0),
        "turn_right": (octagon, -0.05, 0.05, -10.0, -0.2),
        "stopped": (_box(x0, x1, y0, y1), -10.0, 10.0, -10.0, 10.0),  # last = fallback
    }
    return {name: {"points": _poly(p), "linear_min": lo, "linear_max": hi, "theta_min": tlo, "theta_max": thi}
            for name, (p, lo, hi, tlo, thi) in zones.items()}  # fmt: skip


def render_nav2_params(base: str | Path, bundle: str | Path, safety: bool = True, controller: str = "rl") -> dict:
    """Generic ``nav2_params.yaml`` + the bundle's deploy.yaml -> parameters for this robot/policy."""
    deploy = load_deploy(bundle)
    p = copy.deepcopy(yaml.safe_load(Path(base).read_text()))
    ros = deploy.get("ros") or {}
    base_frame, odom_frame = ros.get("base_frame", "base_link"), ros.get("odom_frame", "odom")
    scan = ros.get("scan_topic", "scan")
    fp = _footprint(deploy)
    max_range = float(deploy["lidar"]["max_range"])

    cs = p["controller_server"]["ros__parameters"]
    cs["controller_frequency"] = round(1.0 / float(deploy["control"]["dt"]), 6)
    cs["goal_checker"]["xy_goal_tolerance"] = float(deploy["goal"]["tolerance"])
    fpp = cs["FollowPath"]
    fpp.update(model_dir=str(Path(bundle).resolve()), scan_topic=scan, expected_robot_id=deploy["robot"]["id"],
               safety_state_topic="collision_monitor_state" if safety else "")  # fmt: skip
    if controller == "dwb":
        a = deploy["action"]
        cs["FollowPath"] = {
            "plugin": "dwb_core::DWBLocalPlanner", "trajectory_generator_name": "dwb_plugins::LimitedAccelGenerator",
            "min_vel_x": float(a["v"]["min"]), "max_vel_x": float(a["v"]["max"]),
            "min_vel_y": 0.0, "max_vel_y": 0.0, "max_vel_theta": float(a["w"]["max"]),
            "min_speed_xy": 0.0, "max_speed_xy": float(a["v"]["max"]), "min_speed_theta": 0.0,
            "acc_lim_x": float(a["rate_limit"]["a_max"]), "acc_lim_theta": float(a["rate_limit"]["alpha_max"]),
            "decel_lim_x": -float(a["rate_limit"]["a_max"]),
            "decel_lim_theta": -float(a["rate_limit"]["alpha_max"]), "acc_lim_y": 0.0, "decel_lim_y": 0.0,
            "sim_time": 1.5, "vx_samples": 20, "vy_samples": 1, "vtheta_samples": 40,
            "linear_granularity": 0.025, "angular_granularity": 0.025,
            "critics": ["ObstacleFootprint", "Oscillation", "PathDist", "GoalDist"],
            "PathDist.scale": 16.0, "GoalDist.scale": 24.0, "ObstacleFootprint.scale": 1.0,
        }
    elif controller != "rl":
        raise ValueError("controller must be rl or dwb")
    for name in ("local_costmap", "global_costmap"):
        c = p[name][name]["ros__parameters"]
        c["robot_base_frame"] = base_frame
        c.pop("robot_radius", None)
        c["footprint"] = _poly(fp)
        if name == "local_costmap":
            c["global_frame"] = odom_frame
        c["obstacle_layer"]["scan"].update(topic=scan, raytrace_max_range=max_range,
                                           obstacle_max_range=max_range - 0.5)  # fmt: skip
        c["inflation_layer"]["inflation_radius"] = max(c["inflation_layer"]["inflation_radius"],
                                                       float(deploy["robot"]["footprint_radius"]) + 0.25)  # fmt: skip
    bs = p["behavior_server"]["ros__parameters"]
    bs.update(
        robot_base_frame=base_frame, local_frame=odom_frame, max_rotational_vel=float(deploy["action"]["w"]["max"])
    )
    bt = p["bt_navigator"]["ros__parameters"]
    bt.update(robot_base_frame=base_frame, odom_topic=ros.get("odom_topic", "odom"))
    cm = p["collision_monitor"]["ros__parameters"]
    cm.update(base_frame_id=base_frame, odom_frame_id=odom_frame)
    cm["scan"]["topic"] = scan
    zones = safety_zones(deploy)
    cm["SafetyStop"]["velocity_polygons"] = list(zones)
    for k in [k for k in cm["SafetyStop"] if isinstance(cm["SafetyStop"][k], dict)]:
        del cm["SafetyStop"][k]
    cm["SafetyStop"].update(zones)
    return p


def write_params(params: dict, out: str | Path) -> str:
    Path(out).write_text(yaml.safe_dump(params, sort_keys=False))
    return str(out)
