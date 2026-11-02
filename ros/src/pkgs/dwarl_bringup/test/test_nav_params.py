"""render_nav2_params: generic Nav2 params + a bundle's deploy.yaml -> robot-specific params (no ROS needed)."""

from pathlib import Path

import pytest
import yaml
from dwarl_bringup.nav import load_deploy, render_nav2_params, resolve_bundle, safety_zones

BASE = Path(__file__).parents[1] / "config" / "nav2_params.yaml"
DEPLOY = {
    "version": 2,
    "robot": {
        "id": "rover",
        "drive_type": "diff",
        "footprint_radius": 0.6,
        "footprint": [[0.53, 0.28], [0.53, -0.28], [-0.14, -0.28], [-0.14, 0.28]],
    },
    "control": {"dt": 0.1},
    "lidar": {"max_range": 10.0},
    "goal": {"tolerance": 0.3},
    "action": {"v": {"min": -0.5, "max": 0.5}, "w": {"max": 1.0}, "rate_limit": {"a_max": 1.0, "alpha_max": 2.0}},
    "ros": {"base_frame": "base_link", "odom_frame": "odom", "scan_topic": "/scan", "odom_topic": "/odom"},
}


def _bundle(tmp_path):
    (tmp_path / "deploy.yaml").write_text(yaml.safe_dump(DEPLOY))
    (tmp_path / "policy.onnx").touch()
    return tmp_path


def test_render_uses_robot_frames_footprint_and_rate(tmp_path):
    p = render_nav2_params(BASE, _bundle(tmp_path))
    cs = p["controller_server"]["ros__parameters"]
    assert cs["controller_frequency"] == 10.0 and cs["FollowPath"]["expected_robot_id"] == "rover"
    assert cs["FollowPath"]["scan_topic"] == "/scan" and cs["FollowPath"]["model_dir"] == str(tmp_path.resolve())
    lc = p["local_costmap"]["local_costmap"]["ros__parameters"]
    assert lc["robot_base_frame"] == "base_link" and "0.530" in lc["footprint"]
    assert "robot_radius" not in p["global_costmap"]["global_costmap"]["ros__parameters"]
    cm = p["collision_monitor"]["ros__parameters"]
    assert cm["base_frame_id"] == "base_link" and cm["SafetyStop"]["velocity_polygons"][-1] == "stopped"
    yaml.safe_dump(p)


def test_safety_zones_extend_in_driving_direction():
    z = safety_zones(DEPLOY)
    fast = yaml.safe_load(z["forward_fast"]["points"])
    stopped = yaml.safe_load(z["stopped"]["points"])
    back = yaml.safe_load(z["backward"]["points"])
    assert max(x for x, _ in fast) > max(x for x, _ in stopped)
    assert min(x for x, _ in back) < min(x for x, _ in stopped)


def test_resolve_bundle_explicit_dir(tmp_path):
    assert resolve_bundle(str(_bundle(tmp_path))) == tmp_path


@pytest.mark.parametrize("version,accepted", [(2, True), (3, True), (4, False), (5, True)])
def test_bundle_versions(tmp_path, version, accepted):
    raw = dict(DEPLOY, version=version)
    (tmp_path / "deploy.yaml").write_text(yaml.safe_dump(raw))
    if accepted:
        assert load_deploy(tmp_path)["version"] == version
    else:
        with pytest.raises(RuntimeError, match="unsupported deploy.yaml version 4"):
            load_deploy(tmp_path)


def test_explicit_bundle_and_matched_dwb_limits(tmp_path):
    path = tmp_path / "runs/train/001/exported"
    path.mkdir(parents=True)
    _bundle(path)
    assert resolve_bundle(str(path)) == path
    cs = render_nav2_params(BASE, path, False, "dwb")["controller_server"]["ros__parameters"]
    assert cs["FollowPath"]["plugin"] == "dwb_core::DWBLocalPlanner"
    assert cs["FollowPath"]["max_vel_x"] == DEPLOY["action"]["v"]["max"]
    assert cs["FollowPath"]["acc_lim_theta"] == DEPLOY["action"]["rate_limit"]["alpha_max"]
    assert cs["goal_checker"]["plugin"] == "nav2_controller::SimpleGoalChecker"
    assert cs["goal_checker"]["stateful"] is True


def test_mppi_limits_and_optional_profiling(tmp_path):
    bundle = _bundle(tmp_path)
    plain = render_nav2_params(BASE, bundle, controller="mppi")
    cs = plain["controller_server"]["ros__parameters"]
    mppi = cs["FollowPath"]
    assert mppi["plugin"] == "nav2_mppi_controller::MPPIController"
    assert mppi["motion_model"] == "DiffDrive"
    assert mppi["vx_min"] == DEPLOY["action"]["v"]["min"]
    assert mppi["ax_min"] == -DEPLOY["action"]["rate_limit"]["a_max"]
    assert mppi["model_dt"] == 1 / cs["controller_frequency"]
    for controller in ("dwarl", "dwb", "mppi"):
        base = render_nav2_params(BASE, bundle, controller=controller)
        profiled = render_nav2_params(BASE, bundle, controller=controller, profile_compute=True)
        wrapped = profiled["controller_server"]["ros__parameters"]["FollowPath"]
        assert wrapped["plugin"] == "dwarl_controller::ProfiledController"
        assert wrapped["wrapped_plugin"] == base["controller_server"]["ros__parameters"]["FollowPath"]["plugin"]
        assert base["local_costmap"] == profiled["local_costmap"]


def test_tuning_snapshot_preserves_shared_stack(tmp_path):
    bundle = _bundle(tmp_path)
    tuning = tmp_path / "tuning.yaml"
    tuning.write_text(yaml.safe_dump({"mppi": {"batch_size": 2000, "PathAlignCritic.cost_weight": 8.0}}))
    baseline = render_nav2_params(BASE, bundle, controller="mppi")
    tuned = render_nav2_params(BASE, bundle, controller="mppi", controller_params=tuning, profile_compute=True)
    assert tuned["controller_server"]["ros__parameters"]["FollowPath"]["batch_size"] == 2000
    for section in ("local_costmap", "global_costmap", "collision_monitor", "behavior_server"):
        assert tuned[section] == baseline[section]
    tuning.write_text("mppi:\n  plugin: other\n")
    with pytest.raises(ValueError, match="cannot replace"):
        render_nav2_params(BASE, bundle, controller="mppi", controller_params=tuning)


def test_rotation_envelope_contains_braking_sweep():
    import math

    from dwarl_bringup.nav import _rotation_envelope

    footprint = DEPLOY["robot"]["footprint"]
    for omega in (-1., 1.):
        polygon = _rotation_envelope(footprint, omega, 2., .15, .035)
        # Independently sample intermediate stopping sweeps, including the maximum
        # translational drift in either axis. Every physical corner must stay inside.
        for index in range(1001):
            theta = (omega * .15 + omega * abs(omega) / 4) * index / 1000
            for x, y in footprint:
                for dx, dy in ((-.0325, -.0325), (-.0325, .0325), (.0325, -.0325), (.0325, .0325)):
                    px = math.cos(theta)*x - math.sin(theta)*y + dx
                    py = math.sin(theta)*x + math.cos(theta)*y + dy
                    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                        assert (b[0]-a[0])*(py-a[1]) - (b[1]-a[1])*(px-a[0]) >= -1e-10


def test_shared_safety_tuning_matches_all_controllers(tmp_path):
    bundle = _bundle(tmp_path)
    path = tmp_path / "safety.yaml"
    path.write_text(yaml.safe_dump({"safety": {"rotation_sweep": True},
                                   "progress_checker": {"movement_time_allowance": 8.0},
                                   "dwarl": {"safety_grace": .4}}))
    variants = [render_nav2_params(BASE, bundle, controller=c, controller_params=path)
                for c in ("dwarl", "mppi", "dwb")]
    assert all(v["collision_monitor"] == variants[0]["collision_monitor"] for v in variants)
    assert all(v["controller_server"]["ros__parameters"]["progress_checker"]["movement_time_allowance"] == 8.
               for v in variants)
    assert variants[0]["controller_server"]["ros__parameters"]["FollowPath"]["safety_grace"] == .4


def test_recovery_variant_requires_backup_and_replans(tmp_path):
    import xml.etree.ElementTree as ET

    from dwarl_bringup.nav import render_recovery

    base = BASE.parent.parent / "behavior_trees/dwarl_safety_retry.xml"
    tuning = tmp_path / "tuning.yaml"
    tuning.write_text(yaml.safe_dump({"recovery": {"through_monitor": True, "retry_count": 1,
                                                 "backup_required": True, "replan_after_backup": True}}))
    output, monitor = render_recovery(base, tmp_path / "tree.xml", tuning)
    tree = ET.parse(output)
    assert monitor
    assert tree.find(".//RecoveryNode[@name='SafetyRetry']").get("number_of_retries") == "1"
    sequence = tree.find(".//Sequence[@name='BackUpAndRetry']")
    assert sequence.find("ForceSuccess") is None
    assert sequence.find("Sequence/BackUp") is not None
    assert list(sequence)[-1].tag == "ComputePathToPose"
    assert render_recovery(base, tmp_path / "unused.xml") == (str(base), False)


def test_monitor_only_policy_keeps_external_collision_monitor(tmp_path):
    bundle = _bundle(tmp_path)
    tuning = tmp_path / "monitor_only.yaml"
    tuning.write_text(yaml.safe_dump({"dwarl": {"safety_state_topic": ""}}))
    baseline = render_nav2_params(BASE, bundle, controller="dwarl")
    tuned = render_nav2_params(BASE, bundle, controller="dwarl", controller_params=tuning)
    assert tuned["controller_server"]["ros__parameters"]["FollowPath"]["safety_state_topic"] == ""
    assert tuned["collision_monitor"] == baseline["collision_monitor"]
    assert tuned["collision_monitor"]["ros__parameters"]["SafetyStop"]["action_type"] == "stop"


def test_shared_reverse_limit_cannot_expand_trained_limits(tmp_path):
    bundle = _bundle(tmp_path)
    tuning = tmp_path / "forward.yaml"
    tuning.write_text("motion_limits:\n  min_linear_velocity: 0.0\n")
    for controller, key in (("dwarl", "min_linear_velocity"), ("dwb", "min_vel_x"), ("mppi", "vx_min")):
        params = render_nav2_params(BASE, bundle, controller=controller, controller_params=tuning)
        assert params["controller_server"]["ros__parameters"]["FollowPath"][key] == 0.
    tuning.write_text("motion_limits:\n  min_linear_velocity: -2.0\n")
    with pytest.raises(ValueError, match="trained minimum"):
        render_nav2_params(BASE, bundle, controller="dwarl", controller_params=tuning)


def test_goal_margin_is_shared_and_cannot_relax_scoring(tmp_path):
    bundle = _bundle(tmp_path)
    tuning = tmp_path / "goal.yaml"
    tuning.write_text("goal_checker:\n  xy_goal_tolerance: 0.25\n")
    for controller in ("dwarl", "mppi", "dwb"):
        params = render_nav2_params(BASE, bundle, controller=controller, controller_params=tuning)
        assert params["controller_server"]["ros__parameters"]["goal_checker"]["xy_goal_tolerance"] == .25
    tuning.write_text("goal_checker:\n  xy_goal_tolerance: 0.4\n")
    with pytest.raises(ValueError, match="scored contract"):
        render_nav2_params(BASE, bundle, controller="dwarl", controller_params=tuning)


def test_predictive_approach_retains_stop_guard(tmp_path):
    bundle = _bundle(tmp_path)
    tuning = tmp_path / "approach.yaml"
    tuning.write_text("safety:\n  approach_time: 1.2\n")
    baseline = render_nav2_params(BASE, bundle, controller="dwarl")
    tuned = render_nav2_params(BASE, bundle, controller="dwarl", controller_params=tuning)
    cm = tuned["collision_monitor"]["ros__parameters"]
    assert cm["SafetyStop"] == baseline["collision_monitor"]["ros__parameters"]["SafetyStop"]
    assert cm["polygons"] == ["SafetyStop", "FootprintApproach"]
    assert cm["FootprintApproach"]["action_type"] == "approach"
    assert cm["FootprintApproach"]["time_before_collision"] == 1.2
