"""The training safety layer reproduces the deployed collision monitor's STOP zones and stops the robot."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from matplotlib.path import Path as Polygon
from omegaconf import open_dict

from dwarl.navigation.safety import footprint_polygon, settings, zone_hits

FOOTPRINT = [[0.25, 0.2], [-0.35, 0.2], [-0.35, -0.2], [0.25, -0.2]]
V_MIN, V_MAX, A_MAX = -0.2, 0.8, 0.5


def _bringup_nav():
    path = Path(__file__).resolve().parents[1] / "ros/src/pkgs/dwarl_bringup/dwarl_bringup/nav.py"
    spec = importlib.util.spec_from_file_location("_dwarl_bringup_nav", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _deployed_hits(zones, points, command):
    """Count points inside the first zone whose velocity band holds ``command``, as Nav2's VelocityPolygon does."""
    v, w = command
    for zone in zones.values():
        if zone["linear_min"] <= v <= zone["linear_max"] and zone["theta_min"] <= w <= zone["theta_max"]:
            polygon = np.array(eval(zone["points"]))  # noqa: S307 - our own "[[x, y], ...]" string
            return int(Polygon(polygon).contains_points(points, radius=1e-9).sum())
    raise AssertionError("no zone for command")


@pytest.mark.parametrize("footprint", [FOOTPRINT, None])
def test_zones_match_the_collision_monitor_that_dwarl_bringup_configures(footprint):
    robot = {"footprint": footprint} if footprint else {"footprint_radius": 0.3}
    deploy = {"robot": robot, "action": {"v": {"min": V_MIN, "max": V_MAX}, "w": {"max": 1.5},
                                         "rate_limit": {"a_max": A_MAX, "alpha_max": 2.0}}}
    zones = _bringup_nav().safety_zones(deploy)
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.6, 1.6, (1, 4000, 2))
    polygon = footprint_polygon(footprint, 0.3)
    commands = [(0.0, 0.0), (0.0, 1.0), (0.02, -0.5), (0.1, 0.3), (0.39, 0.0), (0.4, 0.0), (0.8, 0.5),
                (-0.1, 0.0), (0.0, 0.1), (-0.04, 0.25)]
    for command in commands:
        ours = zone_hits(points, np.array([command]), polygon, V_MIN, V_MAX, A_MAX)[0]
        theirs = _deployed_hits(zones, points[0], command)
        # matching up to points on a zone edge (polygon vertices are printed with 3 decimals)
        assert abs(int(ours) - theirs) <= 3, command


def test_settings_are_validated():
    assert settings(None)["enabled"] is False
    with pytest.raises(ValueError, match="unknown"):
        settings({"enable": True})
    with pytest.raises(ValueError, match="min_points"):
        settings({"min_points": 0})


def _env(train_cfg, enabled):
    from dwarl.training.environment.nav_env import NavVecEnv

    train_cfg.env.num_envs = 2
    train_cfg.sim.num_shards = 1
    with open_dict(train_cfg):
        train_cfg.env.safety_layer = {"enabled": enabled}
    return NavVecEnv(train_cfg)


@pytest.mark.parametrize("enabled", [True, False])
def test_an_obstacle_in_the_stop_zone_stops_the_robot_only_when_enabled(train_cfg, enabled):
    env = _env(train_cfg, enabled)
    try:
        sent = []
        env.world.set_ctrl = lambda ctrl: sent.append(np.array(ctrl))
        term = env.drive_action
        # env 0: a wall of returns 5 cm in front of the footprint; env 1: free space
        front = float(env.safety.polygon[:, 0].max()) + 0.05 - env.lidar_origin[0]
        env.ranges[:] = 10.0
        ahead = np.abs(env.lidar.angles) < 0.2
        env.ranges[0, ahead] = front / np.cos(env.lidar.angles[ahead])
        term.command[:] = [0.2, 0.0]
        term.apply_actions()
        assert env.safety.blocked.tolist() == [True, False]
        assert (np.abs(sent[-1][0]) < 1e-12).all() == enabled and (sent[-1][1] > 0).all()
        assert term.command[0].tolist() == ([0.0, 0.0] if enabled else [0.2, 0.0])
        from dwarl.navigation.rewards import safety_zone

        assert safety_zone(env).tolist() == [-1.0, 0.0]
    finally:
        env.close()
