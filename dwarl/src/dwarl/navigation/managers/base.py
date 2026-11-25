"""Configuration and lifecycle shared by managers and stateful MDP terms."""

from importlib import import_module
from inspect import isclass

import numpy as np
from omegaconf import OmegaConf


def config_dict(cfg):
    return OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)


def resolve(value):
    """Resolve an explicit ``package.module:symbol`` or an already callable object."""
    if callable(value):
        return value
    module, sep, name = str(value).partition(":")
    if not sep:
        raise ValueError(f"expected package.module:symbol, got {value!r}")
    result = getattr(import_module(module), name)
    if not callable(result):
        raise TypeError(f"{value!r} is not callable")
    return result


class ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg, self.env = cfg, env

    def reset(self, env_ids):
        """Clear state for just the environments starting a new episode."""


class FunctionTerm:
    """A configured function or stateful callable, both invoked as ``func(env, **params)``."""

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.func = build_function_term(cfg, env)

    def __call__(self, env):
        return np.asarray(self.func(env, **(self.cfg.get("params") or {})))

    def reset(self, env_ids):
        if isinstance(self.func, ManagerTermBase):
            self.func.reset(env_ids)


def build_class_term(cfg, env, base_type):
    """Construct an action/command class, enforcing its lifecycle interface."""
    cls = resolve(cfg["class_type"])
    if not isclass(cls) or not issubclass(cls, base_type):
        raise TypeError(f"{cfg['class_type']!r} must be a {base_type.__name__} subclass")
    return cls(cfg, env)


def build_function_term(cfg, env):
    """Keep pure functions; instantiate stateful ManagerTermBase callables once.

    Both are invoked as ``term(env, **params)``. Classes receive ``(cfg, env)``
    at construction and participate in per-environment resets through the base.
    """
    func = resolve(cfg["func"])
    if not isclass(func):
        return func
    if not issubclass(func, ManagerTermBase):
        raise TypeError(f"{cfg['func']!r} must inherit ManagerTermBase for stateful term resets")
    instance = func(cfg, env)
    if not callable(instance):
        raise TypeError(f"{cfg['func']!r} must implement __call__(env, **params)")
    return instance


def check_batch(value, num_envs, name, scalar=False):
    invalid = value.shape != (num_envs,) if scalar else value.ndim < 2 or value.shape[0] != num_envs
    if invalid:
        raise ValueError(f"term {name!r}: invalid batch shape {value.shape} for {num_envs} environments")
    if not np.isfinite(value).all():
        raise ValueError(f"term {name!r} returned nonfinite values")
