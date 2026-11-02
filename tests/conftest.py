from pathlib import Path

import numpy as np
import pytest
import yaml

from dwarl.datasets.generation.pipeline import generate_dataset


def datagen_cfg():
    cfg = yaml.safe_load((Path(__file__).parents[1] / "src/dwarl/config/recipes/data.yaml").read_text())
    cfg["splits"] = {"train": 4, "val": 2, "test": 4}
    cfg["num_previews"] = 0
    return cfg


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("maps")
    generate_dataset(datagen_cfg(), out, workers=2, log=lambda *_: None)
    return str(out)


@pytest.fixture
def train_cfg(dataset_dir):
    from dwarl.config import compose_config
    from dwarl.training.contract import build_contract

    public = compose_config(assignments=["robot=diffbot", "train.num_envs=4", "train.device=cpu"])
    cfg = build_contract(public)
    cfg.dataset.path = dataset_dir
    cfg.lidar.noise_std, cfg.lidar.dropout_prob = 0., 0.
    return cfg


@pytest.fixture
def rng():
    return np.random.default_rng(0)
