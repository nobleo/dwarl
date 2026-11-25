from importlib.resources import files
from pathlib import Path

import yaml

WORKSPACE = Path(__file__).resolve().parents[1]
TESTBOT = Path(files("dwarl").joinpath("config/fixtures/testbot"))


def datagen_cfg():
    cfg = yaml.safe_load(files("dwarl").joinpath("config/recipes/data.yaml").read_text())
    cfg["splits"] = {"train": 4, "val": 2, "test": 4}
    cfg["num_previews"] = 0
    return cfg


def policy_runner(cfg, dataset, split):
    import torch
    from omegaconf import OmegaConf
    from rsl_rl.runners import OnPolicyRunner

    from dwarl.training.environment.nav_env import NavVecEnv

    cfg.dataset.path, cfg.dataset.split, cfg.env.num_envs = str(dataset), split, 2
    env = NavVecEnv(cfg, split=split, sequential=True)
    runner = OnPolicyRunner(env, OmegaConf.to_container(cfg.agent, resolve=True), log_dir=None)
    # non-trivial normaliser statistics so the in-graph normalisation is actually exercised
    for _ in range(3):
        env.step(torch.rand(2, 2) * 2 - 1)
        runner.alg.actor.update_normalization(env.get_observations())
    return env, runner
