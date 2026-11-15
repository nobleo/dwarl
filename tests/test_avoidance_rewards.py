"""Smoothness and stop-zone rewards that match what the deployed stack does."""

from types import SimpleNamespace

import numpy as np

from dwarl.navigation.rewards import angular_effort, safety_zone

FOOTPRINT = np.array([[0.1, 0.15], [-0.3, 0.15], [-0.3, -0.15], [0.1, -0.15]])


def _env(ranges, speed, angles=None):
    angles = np.linspace(-np.pi, np.pi, len(ranges[0]), endpoint=False) if angles is None else angles
    return SimpleNamespace(
        ranges=np.asarray(ranges, float), lidar=SimpleNamespace(angles=angles), lidar_origin=np.zeros(2),
        footprint=FOOTPRINT, radius=0.34, a_max=1.0,
        reward_context=SimpleNamespace(speed=np.asarray(speed, float)))


def test_stop_zone_grows_with_speed_like_the_collision_monitor():
    angles = np.array([0.0, np.pi / 2, np.pi])
    # an obstacle 0.2 m ahead of the base (0.1 m ahead of the front edge), nothing else close
    ranges = [[0.2, 5.0, 5.0]] * 3
    r = safety_zone(_env(ranges, [0.0, 0.26, 0.5], angles))
    # stopped: 0.1 m > 3.5 cm padding; 0.26 m/s: reach 0.035 + 0.034 + 0.039 = 0.108 m; 0.5 m/s: 0.235 m
    assert r.tolist() == [0.0, -1.0, -1.0]
    side = safety_zone(_env([[5.0, 0.17, 5.0]], [0.0], angles))  # 2 cm beside the footprint
    assert side.tolist() == [-1.0]


def test_angular_effort_penalises_full_angular_acceleration():
    env = SimpleNamespace(alpha_max=2.0, control_dt=0.1, action_manager=SimpleNamespace(
        get_term=lambda name: SimpleNamespace(command_delta=np.array([[0.1, 0.2], [0.0, 0.0], [0.0, -0.1]]))))
    assert np.allclose(angular_effort(env), [-1.0, 0.0, -0.25])
