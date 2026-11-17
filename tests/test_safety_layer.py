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
@pytest.mark.parametrize("v_max", [0.04, 0.08, V_MAX])
def test_zones_match_the_collision_monitor_that_dwarl_bringup_configures(footprint, v_max):
    robot = {"footprint": footprint} if footprint else {"footprint_radius": 0.3}
    deploy = {"robot": robot, "action": {"v": {"min": V_MIN, "max": v_max}, "w": {"max": 1.5},
                                         "rate_limit": {"a_max": A_MAX, "alpha_max": 2.0}}}
    zones = _bringup_nav().safety_zones(deploy)
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.6, 1.6, (1, 4000, 2))
    polygon = footprint_polygon(footprint, 0.3)
    commands = [(0.0, 0.0), (0.0, 1.0), (0.02, -0.5), (0.1, 0.3), (0.39, 0.0), (0.4, 0.0), (0.8, 0.5),
                (-0.1, 0.0), (0.0, 0.1), (-0.04, 0.25), (0.045, 0.3),
                (0.05, 0.3), (-0.05, -0.3), (v_max / 2, 0.3), (0., 0.2), (0., -0.2)]
    for command in commands:
        ours = zone_hits(points, np.array([command]), polygon, V_MIN, v_max, A_MAX)[0]
        theirs = _deployed_hits(zones, points[0], command)
        # matching up to points on a zone edge (polygon vertices are printed with 3 decimals)
        assert abs(int(ours) - theirs) <= 3, command


def test_slow_forward_band_precedes_turn_zone():
    # These side returns are inside the turn octagon but outside the forward box.
    points = np.array([[[0., 0.3], [0.01, 0.3]]])
    polygon = footprint_polygon(FOOTPRINT, 0.3)
    assert zone_hits(points, np.array([[0.045, 0.3]]), polygon, V_MIN, 0.08, A_MAX).item() == 0
    assert zone_hits(points, np.array([[0., 0.3]]), polygon, V_MIN, 0.08, A_MAX).item() == 2


@pytest.mark.parametrize(("command", "point"), [
    ((0.8, 0.), (0.7, 0.)),       # forward braking extension
    ((-0.2, 0.), (-0.43, 0.)),    # reverse braking extension
    ((0., 0.3), (0., 0.3)),      # left rotation
    ((0., -0.3), (0., -0.3)),    # right rotation
])
def test_directional_stop_zones(command, point):
    points = np.array([[point, point]])
    polygon = footprint_polygon(FOOTPRINT, 0.3)
    assert zone_hits(points, np.array([command]), polygon, V_MIN, V_MAX, A_MAX).item() == 2
    assert zone_hits(points, np.array([[0., 0.]]), polygon, V_MIN, V_MAX, A_MAX).item() == 0


def test_settings_are_validated():
    assert settings(None)["enabled"] is False
    with pytest.raises(ValueError, match="unknown"):
        settings({"enable": True})
    with pytest.raises(ValueError, match="min_points"):
        settings({"min_points": 0})


@pytest.mark.parametrize(("key", "value"), [
    ("margin", 0.04), ("latency", 0.2), ("min_points", 3), ("min_points", 2.5),
    ("margin", float("nan")), ("latency", float("inf")),
])
def test_unexported_training_safety_settings_are_rejected(key, value):
    with pytest.raises(ValueError, match=f"safety_layer.{key}.*not exported"):
        settings({key: value})


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
        env.action_manager.action[:] = [0.4, 0.5]
        env.action_manager.prev_action[:] = [0.2, 0.3]
        term.command_delta[:] = [0.1, 0.2]
        history = env.observation_manager.get_term("lidar", "scan")
        env.ranges[:] = 1.
        history(env, history.history, history.normalization)
        old_history = history.buffer.copy()
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
        # Applying STOP must not erase the attempt before reward computation.
        np.testing.assert_allclose(env.action_manager.action[0], [0.4, 0.5])
        np.testing.assert_allclose(env.action_manager.prev_action[0], [0.2, 0.3])
        np.testing.assert_allclose(term.command_delta[0], [0.1, 0.2])
        env.safety.restart_blocked()
        for state, expected in [(env.action_manager.action, [0.4, 0.5]),
                                (env.action_manager.prev_action, [0.2, 0.3]),
                                (term.command_delta, [0.1, 0.2])]:
            np.testing.assert_allclose(state[0], [0., 0.] if enabled else expected)
            np.testing.assert_allclose(state[1], expected)
        # Recovery fills every history slot from the next scan. The unblocked row
        # (and diagnostic-only safety) must retain its older scan history.
        env.ranges[:] = 2.
        history(env, history.history, history.normalization)
        if enabled:
            np.testing.assert_allclose(history.buffer[0], np.repeat(
                history.buffer[0, :1], history.buffer.shape[1], axis=0))
        else:
            np.testing.assert_allclose(history.buffer[0, 1:], old_history[0, :-1])
        np.testing.assert_allclose(history.buffer[1, 1:], old_history[1, :-1])
        from dwarl.navigation.rewards import safety_zone

        assert safety_zone(env).tolist() == [-1.0, 0.0]
    finally:
        env.close()


def test_step_records_blocked_attempt_before_restarting_controller(train_cfg):
    import torch

    from dwarl.navigation.rewards import action_rate, angular_effort

    env = _env(train_cfg, True)
    try:
        env.action_manager.action[:] = [0.1, 0.2]
        env.safety.stops = lambda command: np.array([True, False])
        compute = env.reward_manager.compute
        observe = env._observe
        stages = []

        def reward(dt):
            stages.append("reward")
            np.testing.assert_allclose(env.drive_action.command[0], 0.)
            np.testing.assert_allclose(action_rate(env)[0], -0.18)
            np.testing.assert_allclose(angular_effort(env)[0], -0.25)
            return compute(dt)

        def record(step):
            stages.append("record")
            np.testing.assert_allclose(step.action[0], [0.4, 0.5])
            np.testing.assert_allclose(step.command[0], 0.)
            np.testing.assert_allclose(step.command_delta[0, 1], 0.5 * env.alpha_max * env.control_dt)

        def observation():
            stages.append("observe")
            np.testing.assert_allclose(env.action_manager.action[0], 0.)
            np.testing.assert_allclose(env.action_manager.prev_action[0], 0.)
            np.testing.assert_allclose(env.drive_action.command_delta[0], 0.)
            assert env.observation_manager.get_term("lidar", "scan").fresh[0]
            return observe()

        env.reward_manager.compute = reward
        env.recorder.record_step = record
        env._observe = observation
        env.step(torch.tensor([[0.4, 0.5], [0.4, 0.5]], dtype=torch.float64))
        assert stages == ["reward", "record", "observe"]
    finally:
        env.close()
