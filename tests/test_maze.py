"""Maze generator: variable hallway widths within a single map."""

import numpy as np

from dwarl.datasets.generation.generators import maze
from dwarl.datasets.generation.mapdata import BOX


def _horizontal_wall_ys(prims: np.ndarray, t: float = 0.05) -> np.ndarray:
    """Distinct y-centres of horizontal wall boxes (half-height == wall half-thickness)."""
    boxes = prims[prims[:, 0].astype(int) == BOX]
    horizontal = boxes[np.isclose(boxes[:, 5], t)]
    return np.sort(np.unique(horizontal[:, 2]))


def test_maze_has_variable_corridor_widths():
    rng = np.random.default_rng(0)
    size, t = 8.0, 0.05
    prims = maze(rng, size, corridor=(1.0, 2.0), wall_thickness=2 * t, braid=(0.0, 0.0))
    ys = _horizontal_wall_ys(prims, t)
    assert len(ys) >= 2  # at least two distinct row boundaries
    gaps = np.diff(ys)
    assert gaps.min() > 0
    assert len(np.unique(np.round(gaps, 6))) > 1  # hallway widths vary within the map


def test_maze_uniform_when_corridor_is_single_value():
    rng = np.random.default_rng(1)
    size, t = 8.0, 0.05
    prims = maze(rng, size, corridor=(1.0, 1.0), wall_thickness=2 * t, braid=(0.0, 0.0))
    ys = _horizontal_wall_ys(prims, t)
    gaps = np.diff(ys)
    assert gaps.min() > 0
    assert np.allclose(gaps, gaps[0])  # all hallways equal -> original uniform grid


def test_maze_spans_full_tile():
    rng = np.random.default_rng(2)
    size, t = 8.0, 0.05
    prims = maze(rng, size, corridor=(0.7, 1.5), wall_thickness=2 * t, braid=(0.05, 0.35))
    xs, ys = prims[:, 1], prims[:, 2]
    # walls (grown by the wall half-thickness) must not leak far past the tile edge
    assert np.abs(xs).max() <= size / 2 + t + 1e-6
    assert np.abs(ys).max() <= size / 2 + t + 1e-6


def test_maze_plaza_opens_cells():
    size, t = 8.0, 0.05
    args = dict(corridor=(0.7, 1.0), wall_thickness=2 * t, braid=(0.0, 0.0))
    no_plaza = maze(np.random.default_rng(0), size, plaza=0, **args)
    with_plaza = maze(np.random.default_rng(0), size, plaza=(2, 2), **args)
    # plazas remove walls, so the plaza'd maze has fewer wall boxes
    assert len(with_plaza) < len(no_plaza)
