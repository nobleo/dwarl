"""Portable, hashed robot bundles. Paths in the manifest are bundle-relative."""

import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from omegaconf import open_dict

from dwarl.robot.profile import PROFILE_FILE, ProfileError, dump_yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bundle_xml(xml, profile, out):
    root = ET.fromstring(xml)
    compiler = root.find("compiler")
    description = profile.raw["description"]
    source_dir = profile.uri(description.get("mjcf") or description["urdf"]).parent
    for element in root.findall("./asset/*"):
        filename = element.get("file")
        if not filename:
            continue
        path = Path(filename)
        if not path.is_absolute():
            key = "meshdir" if element.tag == "mesh" else "texturedir"
            directory = compiler.get(key, "") if compiler is not None else ""
            path = source_dir / directory / path
        if not path.is_file():
            raise ProfileError(f"model asset missing: {path}")
        relative = Path("assets") / (digest(path) + path.suffix)
        (out / relative).parent.mkdir(exist_ok=True)
        shutil.copyfile(path, out / relative)
        element.set("file", relative.as_posix())
    if compiler is not None:
        for key in ("meshdir", "texturedir", "assetdir"):
            compiler.attrib.pop(key, None)
    return ET.tostring(root, encoding="unicode")


def write_bundle(out, profile, xml, effective, summary):
    import mujoco

    from dwarl.robot.model import inspect_model

    (out / "robot.xml").write_text(bundle_xml(xml, profile, out))
    inspect_model(mujoco.MjModel.from_xml_path(str(out / "robot.xml")))
    # The frozen profile describes the generated model, not unavailable source URDFs.
    frozen = dict(profile.raw)
    frozen["description"] = {"mjcf": "robot.xml"}
    frozen["limits"] = {key: effective["robot"][key] for key in profile.limits}
    frozen["lidar"] = {**profile.lidar, **effective["lidar"]}
    (out / PROFILE_FILE).write_text(dump_yaml(frozen))
    (out / "source_profile.yaml").write_bytes(profile.path.read_bytes())
    files = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()}
    summary = {key: None if isinstance(value, float) and not math.isfinite(value) else value
               for key, value in summary.items()}
    manifest = {"schema": "dwarl_robot_artifact/1", "model": "robot.xml", "profile": PROFILE_FILE,
                "source": str(profile.path), "source_sha256": profile.sha256,
                "effective": effective, "summary": summary, "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))


def resolve_artifact(cfg, run_dir):
    """Prefer a run's frozen robot even if its original absolute paths still exist."""
    if "robot" not in cfg:
        return cfg
    directory = Path(run_dir) / "robot"
    manifest = directory / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if data.get("schema") != "dwarl_robot_artifact/1":
            raise ProfileError(f"unsupported robot artifact: {manifest}")
        for relative, expected in data["files"].items():
            path = directory / relative
            if not path.is_file() or digest(path) != expected:
                raise ProfileError(f"robot artifact missing or changed: {path}")
    if (directory / "robot.xml").is_file():
        with open_dict(cfg):
            cfg.robot.xml = str((directory / "robot.xml").resolve())
            if (directory / PROFILE_FILE).is_file():
                cfg.robot.profile_file = str((directory / PROFILE_FILE).resolve())
    elif cfg.robot.get("xml") and cfg.robot.get("profile"):
        if not Path(cfg.robot.xml).is_file():
            raise ProfileError(f"saved robot model is missing: {cfg.robot.xml}; restore the run's robot bundle")
    return cfg
