import json
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from dwarl.robot.config import datagen_overrides
from dwarl.runs.store import identity, lock, sha256, write_json

GENERATOR_VERSION = 1


def prepare(public, profile, root):
    data = public.train.data
    if data.path:
        path = Path(data.path)
        index = json.loads((path / "index.json").read_text())
        radius = index.get("config", {}).get("robot_radius")
        if (radius is None or radius + 1e-3 < profile.footprint_radius) and not data.allow_custom:
            raise ValueError("dataset geometry is unspecified or smaller than the robot; "
                             "use train.data.allow_custom=true for an intentional custom dataset")
        return path, sha256(path / "index.json")
    cfg = OmegaConf.load(Path(__file__).parents[1] / "config/recipes/data.yaml")
    cfg = OmegaConf.merge(cfg, datagen_overrides(str(profile.path)))
    cfg.seed = public.train.seed
    cfg.splits = {"train": data.train_maps, "val": data.validation_maps, "test": data.test_maps}
    cfg.planner.pairs_per_map = data.pairs_per_map
    cfg.planner.min_pairs = min(cfg.planner.min_pairs, data.pairs_per_map)
    recipe = OmegaConf.to_container(cfg, resolve=True)
    key = identity({"version": GENERATOR_VERSION, "parameters": recipe})
    root = Path(root).resolve()
    path = root / key
    with lock(root / f".{key}.lock"):
        if not path.exists():
            from dwarl.datasets.generation.pipeline import generate_dataset

            with tempfile.TemporaryDirectory(prefix=".dataset-", dir=root) as temp:
                staged = Path(temp) / "data"
                generate_dataset(recipe, staged, workers=data.workers)
                write_json(staged / "cache.json", {"identity": key, "recipe": recipe})
                staged.rename(path)
        if json.loads((path / "cache.json").read_text())["identity"] != key:
            raise ValueError(f"dataset cache identity mismatch: {path}")
    return path, sha256(path / "index.json")
