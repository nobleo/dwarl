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


def _convex_hull(points):
    points = sorted(set(points))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for target, sequence in ((lower, points), (upper, reversed(points))):
        for point in sequence:
            while len(target) >= 2 and cross(target[-2], target[-1], point) <= 0:
                target.pop()
            target.append(point)
    return lower[:-1] + upper[:-1]


def _rotation_envelope(fp, omega, alpha, latency, margin):
    """Conservative finite stopping sweep for near-in-place rotation (|v| <= .05).

    Includes translation during stopping, footprint margin, and an angular sampling
    bound. This is a model-derived envelope; it is not a hardware braking measurement.
    """
    duration = latency + abs(omega) / alpha
    angle = omega * latency + omega * abs(omega) / (2 * alpha)
    steps = max(1, math.ceil(abs(angle) / .02))
    radius = max(math.hypot(x, y) for x, y in fp)
    padding = margin + .05 * duration + radius * abs(angle) / steps
    points = []
    for index in range(steps + 1):
        theta = angle * index / steps
        c, s = math.cos(theta), math.sin(theta)
        for x, y in fp:
            for dx, dy in ((-padding, -padding), (-padding, padding), (padding, -padding), (padding, padding)):
                points.append((c*x - s*y + dx, s*x + c*y + dy))
    return _convex_hull(points)


def safety_zones(deploy: dict, margin: float = 0.035, latency: float = 0.15, decel: float | None = None,
                 rotation_sweep: bool = False) -> dict:
    """Velocity-dependent collision_monitor STOP zones: footprint bbox + margin, stretched in the driving direction
    by the braking distance v^2 / (2 a) + v * latency at the top speed of each band."""
    fp = _footprint(deploy)
    xs, ys = [p[0] for p in fp], [p[1] for p in fp]
    x0, x1, y0, y1 = min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin
    act = deploy["action"]
    a = float(act["rate_limit"]["a_max"]) if decel is None else decel
    if not all(math.isfinite(v) for v in (margin, latency, a)) or margin < 0 or latency < 0 or a <= 0:
        raise ValueError("safety margin/latency must be nonnegative and deceleration positive")
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
    if rotation_sweep:
        alpha = float(act["rate_limit"]["alpha_max"])
        if alpha <= 0:
            raise ValueError("rotation sweep requires positive angular deceleration")
        omega = float(act["w"]["max"])
        # Keep the existing translation zones and fallback; refine only near-in-place rotation.
        zones["turn_left"] = (_rotation_envelope(fp, omega, alpha, latency, margin),
                              -0.05, 0.05, 0.2, 10.0)
        zones["turn_right"] = (_rotation_envelope(fp, -omega, alpha, latency, margin),
                               -0.05, 0.05, -10.0, -0.2)
    return {name: {"points": _poly(p), "linear_min": lo, "linear_max": hi, "theta_min": tlo, "theta_max": thi}
            for name, (p, lo, hi, tlo, thi) in zones.items()}  # fmt: skip


def render_nav2_params(base: str | Path, bundle: str | Path, safety: bool = True, controller: str = "rl",
                       profile_compute: bool = False, controller_params: str | Path | None = None) -> dict:
    """Generic ``nav2_params.yaml`` + the bundle's deploy.yaml -> parameters for this robot/policy."""
    deploy = load_deploy(bundle)
    p = copy.deepcopy(yaml.safe_load(Path(base).read_text()))
    overrides = {}
    if controller_params:
        overrides = yaml.safe_load(Path(controller_params).read_text())
        allowed = {"mppi", "dwb", "dwarl", "safety", "progress_checker", "recovery", "motion_limits", "goal_checker"}
        if not isinstance(overrides, dict) or set(overrides) - allowed:
            raise ValueError("tuning YAML must contain controller, safety or progress_checker mappings")
        if any(not isinstance(value, dict) for value in overrides.values()):
            raise ValueError("tuning sections must be mappings")
    ros = deploy.get("ros") or {}
    base_frame, odom_frame = ros.get("base_frame", "base_link"), ros.get("odom_frame", "odom")
    scan = ros.get("scan_topic", "scan")
    fp = _footprint(deploy)
    max_range = float(deploy["lidar"]["max_range"])

    cs = p["controller_server"]["ros__parameters"]
    cs["controller_frequency"] = round(1.0 / float(deploy["control"]["dt"]), 6)
    cs["goal_checker"]["xy_goal_tolerance"] = float(deploy["goal"]["tolerance"])
    goal_options = overrides.get("goal_checker", {})
    if set(goal_options) - {"xy_goal_tolerance"}:
        raise ValueError("unsupported goal_checker tuning parameter")
    if "xy_goal_tolerance" in goal_options:
        tolerance = float(goal_options["xy_goal_tolerance"])
        if not 0 < tolerance <= float(deploy["goal"]["tolerance"]):
            raise ValueError("internal goal tolerance must be positive and no looser than the scored contract")
        cs["goal_checker"]["xy_goal_tolerance"] = tolerance
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
    elif controller == "mppi":
        a = deploy["action"]
        cs["FollowPath"] = {
            "plugin": "nav2_mppi_controller::MPPIController", "motion_model": "DiffDrive",
            "time_steps": 56, "model_dt": float(deploy["control"]["dt"]), "batch_size": 1000,
            "iteration_count": 1, "vx_std": 0.2, "vy_std": 0.0, "wz_std": 0.4,
            "vx_max": float(a["v"]["max"]), "vx_min": float(a["v"]["min"]), "vy_max": 0.0,
            "wz_max": float(a["w"]["max"]), "ax_max": float(a["rate_limit"]["a_max"]),
            "ax_min": -float(a["rate_limit"]["a_max"]), "az_max": float(a["rate_limit"]["alpha_max"]),
            "temperature": 0.3, "gamma": 0.015, "visualize": False,
            "critics": ["ConstraintCritic", "CostCritic", "GoalCritic", "GoalAngleCritic",
                        "PathAlignCritic", "PathFollowCritic", "PathAngleCritic", "PreferForwardCritic"],
            "CostCritic.consider_footprint": True,
        }
    elif controller not in ("rl", "dwarl"):
        raise ValueError("controller must be dwarl, mppi or dwb")
    if controller_params:
        selected = overrides.get(controller, {})
        if not isinstance(selected, dict):
            raise ValueError("controller tuning parameters must be a mapping")
        protected = {"plugin", "wrapped_plugin", "profile_compute", "model_dir"}
        if protected & selected.keys():
            raise ValueError("controller tuning cannot replace plugin or profiling configuration")
        cs["FollowPath"].update(selected)
        progress = overrides.get("progress_checker", {})
        if set(progress) - {"required_movement_radius", "movement_time_allowance"}:
            raise ValueError("unsupported progress_checker tuning parameter")
        if any(float(value) <= 0 for value in progress.values()):
            raise ValueError("progress checker values must be positive")
        cs["progress_checker"].update(progress)
    limits = overrides.get("motion_limits", {})
    if set(limits) - {"min_linear_velocity"}:
        raise ValueError("unsupported motion_limits tuning parameter")
    if "min_linear_velocity" in limits:
        minimum = float(limits["min_linear_velocity"])
        if not math.isfinite(minimum) or not float(deploy["action"]["v"]["min"]) <= minimum <= 0:
            raise ValueError("runtime minimum velocity must be between trained minimum and zero")
        key = {"mppi": "vx_min", "dwb": "min_vel_x"}.get(controller, "min_linear_velocity")
        cs["FollowPath"][key] = minimum
    if profile_compute:
        cs["FollowPath"]["profile_compute"] = True
        cs["FollowPath"]["wrapped_plugin"] = cs["FollowPath"]["plugin"]
        cs["FollowPath"]["plugin"] = "dwarl_controller::ProfiledController"
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
    safety_options = dict(overrides.get("safety", {}))
    approach_time = safety_options.pop("approach_time", None)
    zones = safety_zones(deploy, **safety_options)
    cm["SafetyStop"]["velocity_polygons"] = list(zones)
    for k in [k for k in cm["SafetyStop"] if isinstance(cm["SafetyStop"][k], dict)]:
        del cm["SafetyStop"][k]
    cm["SafetyStop"].update(zones)
    if approach_time is not None:
        if not math.isfinite(float(approach_time)) or float(approach_time) <= 0:
            raise ValueError("approach_time must be positive seconds")
        cm["polygons"].append("FootprintApproach")
        cm["FootprintApproach"] = {
            "type": "polygon", "action_type": "approach", "footprint_topic": "/local_costmap/published_footprint",
            "time_before_collision": float(approach_time), "simulation_time_step": 0.05,
            "min_points": 2, "visualize": False, "enabled": True,
        }
    return p


def write_params(params: dict, out: str | Path) -> str:
    Path(out).write_text(yaml.safe_dump(params, sort_keys=False))
    return str(out)


def render_recovery(base, out, tuning=None):
    """Snapshot an explicitly requested recovery variant; default tree is untouched."""
    import xml.etree.ElementTree as ET

    options = (yaml.safe_load(Path(tuning).read_text()) or {}).get("recovery", {}) if tuning else {}
    allowed = {"through_monitor", "retry_count", "backup_required", "replan_after_backup"}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError("unsupported recovery tuning parameter")
    retries = options.get("retry_count", 3)
    if not isinstance(retries, int) or not 0 <= retries <= 6:
        raise ValueError("recovery retry_count must be an integer between 0 and 6")
    for key in allowed - {"retry_count"}:
        if key in options and not isinstance(options[key], bool):
            raise ValueError(f"recovery {key} must be boolean")
    if not options:
        return str(base), False
    tree = ET.parse(base)
    retry = tree.find(".//RecoveryNode[@name='SafetyRetry']")
    retry.set("number_of_retries", str(retries))
    sequence = tree.find(".//Sequence[@name='BackUpAndRetry']")
    if options.get("backup_required"):
        sequence.find("ForceSuccess").tag = "Sequence"
    if options.get("replan_after_backup"):
        ET.SubElement(sequence, "ComputePathToPose", goal="{goal}", path="{path}",
                      planner_id="{selected_planner}", error_code_id="{compute_path_error_code}")
    tree.write(out, encoding="unicode")
    return str(out), options.get("through_monitor", False)
