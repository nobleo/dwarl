"""Demo worlds: valid DWARL datasets whose Nav2 map omits the surprise obstacles."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "demo"))

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


def test_registry_lists_valid_presets():
    registry = yaml.safe_load((Path(__file__).parents[1] / "demo/dwarl_demo/robots.yaml").read_text())
    assert registry["default"] in registry["robots"]
    for preset in registry["presets"].values():
        assert preset["iterations"] > 0 and 0 < preset["validation_interval"] <= preset["iterations"]
