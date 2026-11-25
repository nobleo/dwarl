"""Read-only, Train-only design preview. Run by the opt-in Vite config."""

import json
import sys
from dataclasses import fields, is_dataclass
from tempfile import TemporaryDirectory
from typing import get_args

from omegaconf import OmegaConf

from dwarl.config.compose import compose_config


def describe(node, prefix=""):
    owner = OmegaConf.get_type(node)
    declared = {f.name: f for f in fields(owner)} if is_dataclass(owner) else {}
    for name, value in node.items():
        path = prefix + name
        if not prefix and name not in {"train", "recipe", "robot"} or path == "train.preset":
            continue
        if OmegaConf.is_dict(value) and len(value):
            yield from describe(value, path + ".")
            continue
        field = declared.get(name)
        hint = field.type if field else None
        kind = next((t.__name__ for t in (bool, int, float, str)
                     if hint is t or t in get_args(hint) and type(None) in get_args(hint)), None)
        if kind is None:
            kind = type(value).__name__ if type(value) in (bool, int, float, str) else "json"
        default = OmegaConf.to_container(value) if OmegaConf.is_config(value) else value
        yield path, {"kind": kind, "value": default, "nullable": type(None) in get_args(hint),
                     **(field.metadata if field else {})}


def preview(request):
    with TemporaryDirectory() as cwd:
        preset = request["preset"]
        cfg = compose_config(cwd=cwd, preset=preset)
        catalog = dict(describe(cfg))
        if "edits" not in request:
            return {"fields": catalog}
        assignments = []
        for path, value in request["edits"].items():
            field = catalog[path]
            if "minimum" in field and value < field["minimum"] or "maximum" in field and value > field["maximum"]:
                raise ValueError(f"{path} outside the allowed range")
            assignments.append(f"{path}={json.dumps(value)}")
        resolved = compose_config(cwd=cwd, preset=preset, assignments=assignments)
        return {"resolved": OmegaConf.to_container(resolved, resolve=True)}


if __name__ == "__main__":
    try:
        print(json.dumps(preview(json.load(sys.stdin))))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
