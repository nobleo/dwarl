"""render_nav2_params: generic Nav2 params + a bundle's deploy.yaml -> robot-specific params (no ROS needed)."""

from pathlib import Path

import pytest
import yaml
from dwarl_bringup.nav import load_deploy, render_nav2_params, resolve_bundle, safety_zones

BASE = Path(__file__).parents[1] / "config" / "nav2_params.yaml"
DEPLOY = {
    "version": 2,
    "robot": {
        "id": "cody",
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
    assert cs["controller_frequency"] == 10.0 and cs["FollowPath"]["expected_robot_id"] == "cody"
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
