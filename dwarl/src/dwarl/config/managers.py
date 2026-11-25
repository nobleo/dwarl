"""Typed manager configuration shared by Hydra composition and saved-run loading.

YAML defines the navigation task; these dataclasses define its configuration
contract. Term names and function-specific ``params`` remain extensible.
References stay strings so resolved configs round-trip through ordinary YAML.
"""

from dataclasses import dataclass, field, fields
from math import isfinite
from typing import Any

from omegaconf import MISSING, DictConfig, OmegaConf, open_dict


@dataclass
class TermCfg:
    func: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionTermCfg:
    class_type: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)
    clip: list[float] | None = field(default_factory=lambda: [-1., 1.])


@dataclass
class CommandTermCfg:
    class_type: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationTermCfg(TermCfg):
    clip: list[float] | None = None
    scale: float = 1.


@dataclass
class ObservationGroupCfg:
    concatenate_dim: int = -1
    terms: dict[str, ObservationTermCfg | None] = field(default_factory=dict)


@dataclass
class RewardTermCfg(TermCfg):
    weight: float = MISSING


@dataclass
class TerminationTermCfg(TermCfg):
    time_out: bool = False
    role: str | None = None  # success | collision | stuck; time_out defines timeout semantics


@dataclass
class ManagerBasedNavCfg:
    actions: dict[str, ActionTermCfg | None] = field(default_factory=dict)
    commands: dict[str, CommandTermCfg | None] = field(default_factory=dict)
    observations: dict[str, ObservationGroupCfg | None] = field(default_factory=dict)
    rewards: dict[str, RewardTermCfg | None] = field(default_factory=dict)
    terminations: dict[str, TerminationTermCfg | None] = field(default_factory=dict)


MANAGER_GROUPS = tuple(item.name for item in fields(ManagerBasedNavCfg))


def apply_manager_schemas(cfg: DictConfig) -> DictConfig:
    """Apply typed manager schemas and validate values, returning the same config.

    Update manager groups in place only after all groups pass validation; a failed
    check leaves the input untouched.

    Apply schemas after Hydra's overrides: typed maps otherwise reject adding
    dictionary terms or re-enabling null terms. Validation also covers saved YAML.
    Imports and construction belong to managers; this checks reference syntax
    and configuration values.
    Other job settings and callable-specific parameters have their own owners.
    """
    missing = [name for name in MANAGER_GROUPS if name not in cfg]
    if missing:
        raise ValueError(f"missing manager config groups: {', '.join(missing)}")
    # Merge in the full config context so interpolations may refer to robot/env.
    validated = cfg.copy()
    schema = OmegaConf.structured(ManagerBasedNavCfg)
    with open_dict(validated):
        for name in MANAGER_GROUPS:
            # Rebuild from data so Hydra's force-add flags cannot bypass the schema.
            data = OmegaConf.to_container(cfg[name], resolve=False)
            validated[name] = OmegaConf.merge(schema[name], data)
    for group in MANAGER_GROUPS:
        data = OmegaConf.to_container(validated[group], resolve=True, throw_on_missing=True)
        for name, term in data.items():
            if term is None:
                continue
            path = f"{group}.{name}"
            if group == "observations":
                if not any(value is not None for value in term["terms"].values()):
                    raise ValueError(f"{path}.terms must contain an enabled observation term")
                for key, value in term["terms"].items():
                    if value is not None:
                        _validate_term(value, f"{path}.terms.{key}")
            else:
                _validate_term(term, path)
    with open_dict(cfg):
        for name in MANAGER_GROUPS:
            cfg[name] = validated[name]
    return cfg


def _validate_term(term, path):
    # The applied schema determines which reference field the term carries.
    reference = "class_type" if "class_type" in term else "func"
    if "role" in term:
        if term["role"] not in (None, "success", "collision", "stuck"):
            raise ValueError(f"{path}.role: expected success, collision, stuck or null")
        if term["time_out"] and term["role"] is not None:
            raise ValueError(f"{path}: time_out terms cannot also have an outcome role")
    module, separator, symbol = term[reference].partition(":")
    if not separator or not all(part.isidentifier() for part in module.split(".")) or not symbol.isidentifier():
        raise ValueError(f"{path}.{reference}: expected package.module:symbol")
    clip = term.get("clip")
    if clip is not None and (len(clip) != 2 or not all(isfinite(v) for v in clip) or clip[0] > clip[1]):
        raise ValueError(f"{path}.clip: expected two finite bounds with min <= max")
    for key in ("weight", "scale"):
        if key in term and not isfinite(term[key]):
            raise ValueError(f"{path}.{key}: expected a finite number")
