import importlib.metadata
import platform
from pathlib import Path

from dwarl.runs.store import identity, sha256


def versions():
    values = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("mujoco", "torch", "rsl-rl-lib", "onnx", "onnxruntime", "dwarl-runtime"):
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = None
    return values


def source_identity():
    root = Path(__file__).parents[1]
    return identity({str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
                     if p.suffix in (".py", ".yaml", ".xml")})
