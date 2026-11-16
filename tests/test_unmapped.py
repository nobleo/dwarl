"""Unmapped obstacles: deterministic, on the reference path, and never blocking the route."""

import numpy as np
import pytest

from dwarl.config import compose_config
from dwarl.datasets.generation.geometry import border_primitives, rasterize
from dwarl.datasets.generation.mapdata import BOX, MapData
from dwarl.navigation.clearance import footprint_clearance
from dwarl.testing.scenarios import export_gazebo
from dwarl.testing.unmapped import place_unmapped

ROBOT = {"footprint_radius": 0.3, "footprint": [[0.1, 0.15], [-0.3, 0.15], [-0.3, -0.15], [0.1, -0.15]]}


def _map(walls=()):
    size, res = 10.0, 0.05
    prims = np.concatenate([border_primitives(size, 0.1), np.asarray(walls, np.float32).reshape(-1, 6)])
    n = int(size / res)
    return MapData(occupancy=rasterize(prims, n, res, size), resolution=res, size=size, height=0.5,
                   primitives=prims.astype(np.float32))


def _case(identifier="a"):
    path = np.stack([np.linspace(-4, 4, 161), np.zeros(161)], -1)
    return {"id": identifier, "seed": 42, "start": [-4.0, 0.0, 0.0], "goal": [4.0, 0.0],
            "reference_path": path.tolist()}


def test_obstacles_are_deterministic_on_the_path_and_clear_of_endpoints():
    md, case = _map(), _case()
    first = place_unmapped(md, case, 3, ROBOT, spacing=1.5)
    assert len(first) == 3 and np.array_equal(first, place_unmapped(md, case, 3, ROBOT, spacing=1.5))
    assert not np.array_equal(first, place_unmapped(md, _case("b"), 3, ROBOT, spacing=1.5))
    for prim in first:
        assert abs(prim[2]) < prim[4]  # the obstacle covers the reference path
        assert min(np.hypot(prim[1] + 4, prim[2]), np.hypot(prim[1] - 4, prim[2])) >= 1.5
        assert footprint_clearance(np.array(case["start"][:2]), 0.0, 0.3, np.array(ROBOT["footprint"]),
                                   prim[None]) > 0  # fmt: skip


def test_obstacles_never_close_a_narrow_corridor():
    walls = [[BOX, 0.0, 0.5, 0.0, 5.0, 0.05], [BOX, 0.0, -0.5, 0.0, 5.0, 0.05]]
    assert len(place_unmapped(_map(walls), _case(), 3, ROBOT)) == 0  # 0.9 m corridor: no room for a detour
    walls = [[BOX, 0.0, 1.0, 0.0, 5.0, 0.05], [BOX, 0.0, -1.0, 0.0, 5.0, 0.05]]
    assert len(place_unmapped(_map(walls), _case(), 3, ROBOT)) > 0  # 1.9 m corridor: room to pass


def test_unmapped_obstacles_are_in_the_world_only(tmp_path):
    md = _map()
    obstacles = place_unmapped(md, _case(), 2, ROBOT)
    export_gazebo(md, tmp_path, "world", obstacles)
    export_gazebo(md, tmp_path / "plain", "world")
    shown, plain = ((p / "world.sdf").read_text().count("<collision") for p in (tmp_path, tmp_path / "plain"))
    assert len(obstacles) == 2 and shown - plain == 2
    assert (tmp_path / "world.pgm").read_bytes() == (tmp_path / "plain" / "world.pgm").read_bytes()


def test_unmapped_obstacles_and_planner_are_validated():
    with pytest.raises(ValueError, match="gazebo"):
        compose_config(assignments=["test.unmapped_obstacles=2"])
    with pytest.raises(ValueError, match="planner"):
        compose_config(assignments=["test.sim=gazebo", "test.planner=astar"])
    cfg = compose_config(assignments=["test.sim=gazebo", "test.planner=smac2d", "test.unmapped_obstacles=2"])
    assert cfg.test.planner == "smac2d" and cfg.test.unmapped_obstacles == 2
