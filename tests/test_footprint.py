"""Footprint-aware (rectangle) inflation for the axis-aligned maze/rooms generators.

A rectangular footprint traverses axis-aligned passages at its lateral half-width
(``passage_radius``), not its circumscribed radius, so mazes for wide robots get genuinely tight
corridors while each planned path is still validated against the footprint's inscribed width.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from dwarl.datasets.generation.mapdata import BOX
from dwarl.datasets.generation.pipeline import generate_map
from dwarl.navigation.clearance import path_width_clear
from dwarl.robot.config import datagen_overrides

from .conftest import datagen_cfg

TESTBOT = Path(__file__).parent / "robots" / "testbot" / "dwarl_robot.yaml"
FOOTPRINT = [[0.21, 0.16], [0.21, -0.16], [-0.1, -0.16], [-0.1, 0.16]]
RADIUS = 0.264  # circumscribed radius (hypot(0.21, 0.16))
PASSAGE = 0.16  # lateral half-width


def _wall(x0, y0, x1, y1, t=0.1):
    return [BOX, (x0 + x1) / 2, (y0 + y1) / 2, 0.0, max(abs(x1 - x0) / 2, t / 2), max(abs(y1 - y0) / 2, t / 2)]


def _min_free_passage(prims: np.ndarray, wall_thickness: float = 0.1) -> float:
    """Smallest free gap between parallel interior maze walls (cell width minus wall thickness)."""
    boxes = prims[prims[:, 0].astype(int) == BOX]
    t = wall_thickness / 2
    horiz = boxes[np.isclose(boxes[:, 5], t)]
    vert = boxes[np.isclose(boxes[:, 4], t)]

    def gap(centers):
        c = np.sort(np.unique(centers))
        return np.inf if len(c) < 2 else float(np.diff(c).min())

    return min(gap(horiz[:, 2]), gap(vert[:, 1])) - wall_thickness


def test_datagen_overrides_rect_footprint():
    o = datagen_overrides(str(TESTBOT))
    assert o["robot_radius"] == pytest.approx(RADIUS, abs=1e-3)
    assert o["passage_radius"] == pytest.approx(PASSAGE, abs=1e-3)
    assert o["footprint"] == FOOTPRINT


def test_datagen_overrides_circular_footprint():
    o = datagen_overrides("diffbot")
    assert o["robot_radius"] == pytest.approx(0.18, abs=1e-3)
    assert "footprint" not in o and "passage_radius" not in o


def test_path_width_clear_straight_corridor():
    path = np.array([[-3.0, 0.0], [3.0, 0.0]])
    # free corridor width 0.40 m > footprint width 0.32 m -> clear
    wide = np.asarray([_wall(-4, 0.25, 4, 0.25), _wall(-4, -0.25, 4, -0.25)], np.float32).reshape(-1, 6)
    assert path_width_clear(path, FOOTPRINT, wide)
    # free corridor width 0.20 m < footprint width 0.32 m -> blocked
    narrow = np.asarray([_wall(-4, 0.15, 4, 0.15), _wall(-4, -0.15, 4, -0.15)], np.float32).reshape(-1, 6)
    assert not path_width_clear(path, FOOTPRINT, narrow)


def test_path_width_clear_empty_path():
    assert path_width_clear(np.zeros((1, 2)), FOOTPRINT, np.zeros((0, 6)))


@pytest.mark.parametrize("kind,yaw", [(0, 0.), (0, .7), (1, 0.)])
def test_path_width_checks_between_waypoints(kind, yaw):
    obstacle = np.array([[kind, 0., 0., yaw, .1, .1]])
    path = np.array([[-1., 0.], [1., 0.]])
    assert not path_width_clear(path, FOOTPRINT, obstacle)
    assert path_width_clear(path + [0., .5], FOOTPRINT, obstacle)
    assert not path_width_clear(np.array([[0., 0.]]), FOOTPRINT, obstacle)


def test_maze_footprint_corridors_track_half_width():
    cfg = datagen_cfg()
    cfg["generators"] = {"maze": cfg["generators"]["maze"]}
    cfg["footprint"] = FOOTPRINT
    cfg["passage_radius"] = PASSAGE
    cfg["robot_radius"] = RADIUS
    md = next(m for m in (generate_map(cfg, s) for s in range(20)) if m is not None)
    assert md is not None
    margin = float(cfg["inflation_margin"])
    wall = float(cfg["generators"]["maze"]["wall_thickness"])
    passage = _min_free_passage(md.primitives[4:], wall)  # strip the 4 border walls
    # tighter than the circumscribed radius would allow...
    assert passage < 2 * (RADIUS + margin)
    # ...but never below the robot's lateral-width floor
    assert passage >= 2 * (PASSAGE + margin) - 1e-6
    # plazas provide an open area large enough for a random-heading spawn (which tight corridors cannot)
    clearance = ndimage.distance_transform_edt(~md.occupancy) * md.resolution
    assert (clearance >= RADIUS + float(cfg["spawn"]["extra_clearance"])).any()
    for k in range(md.num_pairs):
        assert path_width_clear(md.path(k), FOOTPRINT, md.primitives)
