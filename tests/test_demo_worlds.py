"""Demo worlds: valid DWARL datasets whose Nav2 map omits the surprise obstacles."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "demo"))

from dwarl_demo.__main__ import (  # noqa: E402
    CONTROLLERS,
    controllers_for,
    default_robot,
    load_registry,
    planner_for,
    unmapped_for,
)
from dwarl_demo.navigate import surprise_scenes, unmapped_scenes  # noqa: E402
from dwarl_demo.worlds import WORLDS, build_worlds  # noqa: E402

from dwarl.datasets.generation.mapdata import MapData  # noqa: E402
from dwarl.training.environment.dataset import MapDataset  # noqa: E402

TESTBOT = Path(__file__).parent / "robots" / "testbot"


@pytest.mark.parametrize("robot", [None, "diffbot", TESTBOT])
def test_demo_worlds_are_scored_datasets(tmp_path, robot):
    out = build_worlds(tmp_path / "worlds", robot, previews=False)
    dataset = MapDataset(out, "scenarios")
    assert sorted(m.generator for m in dataset.maps) == sorted(WORLDS)
    for name in WORLDS:
        md = MapData.load(out / "scenarios" / f"{name}.npz")
        assert md.num_pairs == 2 and all(len(md.path(k)) > 50 for k in range(2))
        sdf = (out / "gazebo" / f"{name}.sdf").read_text()
        shown = (out / "gazebo" / f"{name}_surprise.sdf").read_text()
        assert shown.count("<collision") > sdf.count("<collision")  # surprises only in Gazebo
        assert yaml.safe_load((out / "gazebo" / f"{name}_surprise.yaml").read_text())["image"] == f"{name}.pgm"
        pairs = json.loads((out / "gazebo" / f"{name}_surprise.json").read_text())["pairs"]
        assert np.allclose(pairs[0]["start"][:2], md.path(0)[0])


def test_registry_lists_valid_presets(tmp_path):
    registry = load_registry(tmp_path)
    assert default_robot(registry["robots"]) in registry["robots"]
    for preset in registry["presets"].values():
        assert preset["iterations"] > 0 and 0 < preset["validation_interval"] <= preset["iterations"]


def test_local_registry_can_add_the_default_robot(tmp_path):
    builtin = load_registry(tmp_path)["robots"]
    (tmp_path / "robots.yaml").write_text(yaml.safe_dump(
        {"robots": {"my_robot": {"profile": "my_robot_dwarl", "default": True}}}))
    robots = load_registry(tmp_path)["robots"]
    assert set(robots) == set(builtin) | {"my_robot"}
    assert default_robot(robots) == "my_robot"
    assert default_robot({"a": {}, "b": {}}) == "a"


def test_comparison_settings_default_to_the_shared_setup():
    entry = load_registry(Path("/nonexistent"))["robots"]["turtlebot3"]
    assert controllers_for(entry) == CONTROLLERS and "mppi" in CONTROLLERS
    assert planner_for(entry) == "smac2d"
    assert unmapped_for(entry, {}) == 3
    assert unmapped_for(entry, {"unmapped_obstacles": 1}) == 1
    assert unmapped_for({"unmapped_obstacles": 0}, {"unmapped_obstacles": 1}) == 0


def test_showcase_scenes(tmp_path):
    out = build_worlds(tmp_path / "worlds", None, previews=False)
    scenes = surprise_scenes(out)
    assert [s["world"] for s in scenes] == sorted(WORLDS)
    assert all(s["sdf"].is_file() and s["map"].is_file() and s["path_length"] > 0 for s in scenes)
    assert unmapped_scenes(tmp_path / "no-run") == []
