"""Large layouts use the same dataset and Gazebo export as every other benchmark."""

import json

import numpy as np
import pytest
import yaml
from tests.support import TESTBOT

from dwarl.datasets.generation.mapdata import MapData
from dwarl.testing.scenarios import build_scenarios, export_gazebo


@pytest.mark.parametrize("robot", [None, "diffbot", TESTBOT])
def test_large_worlds_are_standard_benchmarks(tmp_path, robot):
    names = ["office", "warehouse"]
    build_scenarios(tmp_path / "worlds", names=names, robot=robot, previews=False)
    out = tmp_path / "worlds"
    for name in names:
        md = MapData.load(out / "scenarios" / f"{name}.npz")
        assert md.num_pairs == 2 and all(len(md.path(k)) > 50 for k in range(2))
        pairs = json.loads((out / "gazebo" / f"{name}.json").read_text())["pairs"]
        assert np.allclose(pairs[0]["start"][:2], md.path(0)[0])
        assert yaml.safe_load((out / "gazebo" / f"{name}.yaml").read_text())["image"] == f"{name}.pgm"
        # An obstacle changes only the simulated scene; the planner still sees the clean map.
        export_gazebo(md, out / "obstructed", name, np.array([[1, 0, 0, 0, .2, .2]]))
        assert (out / "obstructed" / f"{name}.sdf").read_text().count("<collision") > (
            out / "gazebo" / f"{name}.sdf").read_text().count("<collision")
        assert (out / "obstructed" / f"{name}.pgm").read_bytes() == (out / "gazebo" / f"{name}.pgm").read_bytes()
