"""Unmapped obstacles: deterministic, on the reference path, and never blocking the route."""

import numpy as np
import pytest

from dwarl.config import compose_config
from dwarl.datasets.generation.geometry import border_primitives, rasterize
from dwarl.datasets.generation.mapdata import BOX, MapData
from dwarl.navigation.clearance import footprint_clearance
from dwarl.testing.scenarios import export_gazebo
from dwarl.testing.unmapped import narrow_points, place_unmapped

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
    with pytest.raises(ValueError, match="gazebo or --sim mujoco"):
        compose_config(assignments=["test.sim=both", "test.unmapped_obstacles=2"])
    assert compose_config(assignments=["test.unmapped_obstacles=2"]).test.unmapped_obstacles == 2  # MuJoCo
    with pytest.raises(ValueError, match="planner"):
        compose_config(assignments=["test.sim=gazebo", "test.planner=astar"])
    cfg = compose_config(assignments=["test.sim=gazebo", "test.planner=smac2d", "test.unmapped_obstacles=2"])
    assert cfg.test.planner == "smac2d" and cfg.test.unmapped_obstacles == 2


def test_obstacles_keep_clear_of_doorways():
    # A wall across the room with a 1 m door at x=0; the path runs through the door.
    walls = [[BOX, 0.0, 2.75, 0.0, 0.05, 2.25], [BOX, 0.0, -2.75, 0.0, 0.05, 2.25]]
    md = _map(walls)
    narrow = narrow_points(md, np.asarray(_case()["reference_path"]), ROBOT)
    assert len(narrow) and np.abs(narrow[:, 0]).max() < 0.6  # only the doorway is narrow
    for identifier in "abcdef":
        for prim in place_unmapped(md, _case(identifier), 3, ROBOT, spacing=1.5):
            assert abs(prim[1]) >= 2 * ROBOT["footprint_radius"] + prim[4] - 1e-6


def test_doorway_option_also_places_obstacles_behind_doors():
    walls = [[BOX, 0.0, 2.75, 0.0, 0.05, 2.25], [BOX, 0.0, -2.75, 0.0, 0.05, 2.25]]
    md = _map(walls)
    near = []
    for identifier in "abcdefghijklmnopqrst":
        aware = place_unmapped(md, _case(identifier), 1, ROBOT)
        anywhere = place_unmapped(md, _case(identifier), 1, ROBOT, doorways=True)
        assert len(anywhere) >= len(aware)
        near += [p for p in anywhere if abs(p[1]) < 2 * ROBOT["footprint_radius"] + p[4]]
    assert near  # some obstacles now sit close to the door
    cfg = compose_config(assignments=["test.sim=gazebo", "test.unmapped_obstacles=3",
                                      "test.unmapped_near_doorways=true"])
    assert cfg.test.unmapped_near_doorways


def test_doorway_exclusion_only_removes_obstacles_without_moving_the_rest():
    walls = [[BOX, 0.0, 2.75, 0.0, 0.05, 2.25], [BOX, 0.0, -2.75, 0.0, 0.05, 2.25]]
    md = _map(walls)
    removed = retained = 0
    for identifier in "abcdefghijklmnopqrst":
        case = _case(identifier)
        anywhere = place_unmapped(md, case, 3, ROBOT, spacing=1.5, doorways=True)
        narrow = narrow_points(md, np.asarray(case["reference_path"]), ROBOT)
        keep = [p for p in anywhere
                if np.linalg.norm(narrow - p[1:3], axis=1).min() >= 2 * ROBOT["footprint_radius"] + p[4]]
        expected = np.asarray(keep, np.float32).reshape(-1, 6)
        np.testing.assert_array_equal(place_unmapped(md, case, 3, ROBOT, spacing=1.5), expected)
        removed += len(anywhere) - len(expected)
        retained += len(expected)
    assert removed and retained


def test_missing_recorded_route_fails_before_starting_gazebo(tmp_path):
    from omegaconf import OmegaConf

    from dwarl.testing.backends.gazebo import GazeboBackend

    backend = GazeboBackend.__new__(GazeboBackend)
    backend.directory, backend.controller = tmp_path, "dwarl"
    backend.contract = OmegaConf.create({"robot": ROBOT})
    backend.request = OmegaConf.create({"unmapped_route": str(tmp_path / "prior"), "unmapped_obstacles": 3})
    backend._routes = {"another-case": [[-4, 0], [4, 0]]}
    _map().save(tmp_path / "map.npz")
    with pytest.raises(ValueError, match="unmapped_route:.*selected case a"):
        backend.execute_case(dict(_case(), map_file="map.npz"))


def test_obstacles_can_follow_nav2s_recorded_route(tmp_path):
    import json

    from dwarl.testing.unmapped import recorded_routes

    plan = [[-4.0, 0.0], *[[x, 2.0] for x in np.linspace(-3, 3, 61)], [4.0, 0.0]]  # another "aisle"
    rows = [{"case_id": "a", "unmapped_obstacles": [], "diagnostics": {"global_plans": [plan, plan[:3]]}},
            {"case_id": "b", "unmapped_obstacles": [[1, 0, 0, 0, .2, .2]], "diagnostics": {"global_plans": [plan]}}]
    (tmp_path / "episodes.json").write_text(json.dumps({"gazebo-mppi": rows}))
    routes = recorded_routes(tmp_path)
    assert list(routes) == ["a"] and routes["a"][1] == [-3.0, 2.0]  # first plan; runs with obstacles ignored
    md = _map()
    for prim in place_unmapped(md, dict(_case(), reference_path=routes["a"]), 2, ROBOT, spacing=1.5):
        assert abs(prim[2] - 2.0) < prim[4]  # on Nav2's route, not on the dataset path at y = 0
    with pytest.raises(ValueError, match="unmapped_route"):
        compose_config(assignments=["test.sim=mujoco", "test.unmapped_obstacles=2",
                                    f"test.unmapped_route={tmp_path}"])


def test_trajectory_plot_shows_nav2s_first_plan(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dwarl.testing.results import trajectory_plot

    case = dict(_case(), name="corridor")
    row = {"outcome": "success", "simulator": "gazebo", "trajectory": [[-4, 0], [4, 0]],
           "diagnostics": {"global_plans": [[[-4, 0, 0], [0, 1, 0], [4, 0, 0]]]}}
    trajectory_plot(plt, _map(), case, row, {}, tmp_path / "t.png")
    assert (tmp_path / "t.png").stat().st_size > 0
