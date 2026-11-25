"""Describe the workspace's composed settings without starting a workflow."""

from dataclasses import fields, is_dataclass
from threading import Lock
from typing import get_args

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from dwarl.config.compose import compose_config

_compose_lock = Lock()  # Hydra's initialization context is process-global.
_dedicated = {"train.preset", "train.resume", "train.detached", "test.run", "test.checkpoint",
              "deploy.run", "deploy.checkpoint"}


def editable(workflow, path):
    if workflow == "datagen":
        return path == "train.seed" or path.startswith("train.data.")
    return (path not in _dedicated and
            (path.startswith(workflow + ".") or workflow == "train" and path.startswith("recipe.")))


def _describe(node, prefix, workflow):
    owner = OmegaConf.get_type(node)
    declared = {f.name: f for f in fields(owner)} if is_dataclass(owner) else {}
    for name, value in node.items():
        path = prefix + name
        if not editable(workflow, path) and not (workflow == "datagen" and path == "train.data"):
            continue
        if OmegaConf.is_dict(value) and len(value):
            yield from _describe(value, path + ".", workflow)
            continue
        field = declared.get(name)
        hint = field.type if field else None
        optional = type(None) in get_args(hint)
        kind = next((t.__name__ for t in (bool, int, float, str)
                     if hint is t or optional and t in get_args(hint)), None)
        if not kind:
            kind = type(value).__name__ if type(value) in (bool, int, float, str) else "json"
        yield path, {"kind": kind, "nullable": optional, "value":
                     OmegaConf.to_container(value) if OmegaConf.is_config(value) else value,
                     **(field.metadata if field else {})}


def catalog(root, workflow, preset):
    if workflow not in ("datagen", "train", "test", "deploy"):
        raise ValueError("unknown settings workflow")
    try:
        with _compose_lock:
            cfg = compose_config(cwd=root, preset=preset or None)
            groups = {"train": ("train", "recipe"), "datagen": ("train",)}.get(workflow, (workflow,))
            return dict(row for group in groups for row in _describe(cfg[group], group + ".", workflow))
    except OmegaConfBaseException as exc:
        raise ValueError(str(exc)) from exc
