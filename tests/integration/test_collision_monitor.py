"""The training safety layer reproduces the deployed collision monitor's STOP zones and stops the robot."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from matplotlib.path import Path as Polygon

from dwarl.navigation.safety import footprint_polygon, zone_hits

FOOTPRINT = [[0.25, 0.2], [-0.35, 0.2], [-0.35, -0.2], [0.25, -0.2]]
V_MIN, V_MAX, A_MAX = -0.2, 0.8, 0.5


def _bringup_nav():
    path = Path(__file__).resolve().parents[2] / "ros/src/pkgs/dwarl_bringup/dwarl_bringup/nav.py"
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
