from difflib import get_close_matches
from math import isfinite
from pathlib import Path

from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from .schema import Config

PATHS = ("robot", "train.resume", "train.data.path", "test.run", "test.dataset", "test.controller_params", "deploy.run")


def _paths(cfg, base):
    for key in PATHS:
        value = OmegaConf.select(cfg, key)
        if value is None or value == "latest":
            continue
        if key == "robot" and "/" not in value and not (base / value).exists():
            continue  # builtin or explicit ROS package reference
        OmegaConf.update(cfg, key, str((base / Path(value).expanduser()).resolve()))


def _known(cfg, prefix=""):
    for key, value in cfg.items():
        path = prefix + key
        yield path
        if OmegaConf.is_dict(value):
            yield from _known(value, path + ".")


def validate_training(train):
    for name in ("validation_interval", "benchmark_interval"):
        interval = train[name]
        if interval < 0:
            raise ValueError(f"train.{name} must be nonnegative (0 disables it)")
        if interval and interval % train.save_interval:
            raise ValueError(f"train.{name} must be a multiple of train.save_interval")
    if train.benchmark_cases <= 0 or train.early_stop.patience <= 0:
        raise ValueError("train.benchmark_cases and train.early_stop.patience must be positive")
    if not isfinite(train.early_stop.min_delta) or not 0 <= train.early_stop.min_delta <= 1:
        raise ValueError("train.early_stop.min_delta must be between 0 and 1")
    if train.early_stop.enabled and not train.validation_interval:
        raise ValueError("early stopping requires train.validation_interval > 0")


def compose_config(config=None, assignments=(), cwd=None, explicit_keys=None, preset=None):
    cwd = Path(cwd or Path.cwd()).resolve()
    ConfigStore.instance().store(name="dwarl_public", node=Config)
    with initialize_config_module("dwarl.config.recipes", version_base="1.3"):
        cfg = compose(config_name="public")
    if preset:
        from .presets import PRESETS

        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose {', '.join(PRESETS)}")
        cfg = OmegaConf.merge(cfg, {"train": PRESETS[preset]})
    source = (cwd / config).resolve() if config else cwd / "dwarl.yaml"
    if config or source.exists():
        supplied = OmegaConf.load(source)
        _paths(supplied, source.parent)
        if explicit_keys is not None:
            explicit_keys.extend(k for k in _known(supplied) if not OmegaConf.is_dict(OmegaConf.select(supplied, k)))
        cfg = OmegaConf.merge(cfg, supplied)
    seen = set()
    for assignment in assignments:
        key, sep, value = assignment.partition("=")
        if not sep:
            raise ValueError(f"expected schema.path=value, got {assignment!r}")
        if any(key == other or key.startswith(other + ".") or other.startswith(key + ".") for other in seen):
            raise ValueError(f"duplicate CLI assignment: {key}")
        seen.add(key)
        if explicit_keys is not None:
            explicit_keys.append(key)
        if key not in set(_known(cfg)):
            suggestions = get_close_matches(key, list(_known(cfg)), n=3)
            raise ValueError(f"unknown key {key!r}; suggestions: {', '.join(suggestions)}")
        update = OmegaConf.from_dotlist([assignment])
        _paths(update, cwd)
        cfg = OmegaConf.merge(cfg, update)
    OmegaConf.resolve(cfg)
    if cfg.schema != "dwarl/1":
        raise ValueError(f"unsupported schema: {cfg.schema}")
    for key in ("train.num_envs", "train.max_iterations", "train.save_interval", "train.validation_cases",
                "train.data.train_maps", "train.data.validation_maps", "train.data.test_maps",
                "train.data.pairs_per_map", "train.data.workers", "test.repeats",
                "test.startup_timeout", "test.stall_timeout"):
        if OmegaConf.select(cfg, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg.test.max_cases is not None and cfg.test.max_cases <= 0:
        raise ValueError("test.max_cases must be positive")
    validate_training(cfg.train)
    if cfg.test.sim not in ("mujoco", "gazebo", "both"):
        raise ValueError("test.sim must be mujoco, gazebo or both")
    controllers = list(cfg.test.controllers)
    if not controllers or len(set(controllers)) != len(controllers) or any(
        c not in ("dwarl", "mppi", "dwb", "rpp") for c in controllers
    ):
        raise ValueError("test.controllers must contain unique choices: dwarl, mppi, dwb, rpp")
    if controllers != ["dwarl"] and cfg.test.sim != "gazebo":
        raise ValueError("MPPI/DWB/RPP comparisons require --sim gazebo")
    if cfg.test.profile_compute and cfg.test.sim != "gazebo":
        raise ValueError("compute profiling requires --sim gazebo")
    if cfg.test.controller_params and cfg.test.sim != "gazebo":
        raise ValueError("controller tuning requires --sim gazebo")
    if cfg.test.suite not in ("benchmark", "validation", "test"):
        raise ValueError("test.suite must be benchmark, validation or test")
    for key in ("recipe.env.control_dt", "recipe.env.episodes_per_map", "recipe.sim.timestep",
                "recipe.sim.pool_size", "recipe.agent.num_steps_per_env", "recipe.agent.algorithm.num_mini_batches",
                "recipe.agent.algorithm.num_learning_epochs", "recipe.agent.algorithm.learning_rate"):
        if OmegaConf.select(cfg, key) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in _known(cfg):
        value = OmegaConf.select(cfg, key)
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(f"{key} must be finite")
    from dwarl.config.managers import apply_manager_schemas

    apply_manager_schemas(cfg.recipe)
    return cfg
