"""Resolve a run config's robot profile into concrete training values (``materialize``).

``train.yaml`` only names the profile (``robot.profile``); limits / lidar keys left ``null`` are filled
from it, explicitly set keys win (e.g. ``lidar.noise_std=0.02`` for an ablation). The generated MuJoCo
model and a copy of the profile are written next to the run, so eval/export/play never need the robot's
repository (or ROS) again.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict

from dwarl.robot.build import build_xml, summarize
from dwarl.robot.profile import PROFILE_FILE, load_profile
from dwarl.robot.schema import validate_measurements

LIMIT_KEYS = ("v_min", "v_max", "w_max", "a_max", "alpha_max")
LIDAR_KEYS = ("num_rays", "fov_deg", "min_range", "max_range", "noise_std", "dropout_prob")


def is_materialized(cfg: DictConfig) -> bool:
    xml = OmegaConf.select(cfg, "robot.xml", default=None)
    return bool(xml) and Path(xml).is_file() and OmegaConf.select(cfg, "robot.footprint_radius") is not None


def materialize(cfg: DictConfig, out_dir: str | Path) -> dict[str, Any]:
    """Fill ``cfg.robot`` / ``cfg.lidar`` (and the lidar encoder padding) from the profile, in place."""
    from dwarl.robot.artifact import write_bundle

    out = Path(out_dir).resolve()
    p = load_profile(cfg.robot.profile)
    xml = build_xml(p)
    s = summarize(p, xml)
    resolved = deepcopy(cfg)
    apply_profile(resolved, p, s, out)
    validate_effective(resolved)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"robot artifact destination is not empty: {out}")
    with tempfile.TemporaryDirectory(prefix=".robot-", dir=out.parent) as tmp:
        staging = Path(tmp) / "bundle"
        staging.mkdir()
        effective = OmegaConf.to_container(OmegaConf.create({"robot": resolved.robot, "lidar": resolved.lidar}),
                                          resolve=True)
        # Runtime paths are resolved by the run loader; persisted metadata uses relative paths.
        effective["robot"]["xml"] = "robot.xml"
        effective["robot"]["profile_file"] = PROFILE_FILE
        write_bundle(staging, p, xml, effective, s.as_dict())
        if out.exists():
            out.rmdir()
        shutil.move(str(staging), out)
    apply_profile(cfg, p, s, out)
    validate_effective(cfg)
    return s.as_dict()


def validate_effective(cfg):
    validate_measurements({k: cfg.robot[k] for k in LIMIT_KEYS}, {k: cfg.lidar[k] for k in LIDAR_KEYS})
    full = math.isclose(float(cfg.lidar.fov_deg), 360.)
    for role in ("actor", "critic"):
        enc = OmegaConf.select(cfg, f"agent.{role}.encoder_cfg", default=None)
        if enc is not None:
            with open_dict(enc):
                enc.circular = full


def apply_profile(cfg, p, s, out):
    """Apply resolved profile settings without performing filesystem operations."""
    with open_dict(cfg):
        rc = cfg.robot
        for k in LIMIT_KEYS:
            if rc.get(k) is None:
                rc[k] = p.limits[k]
        rc.id = p.id
        rc.xml = str((out / "robot.xml").resolve())
        rc.profile_file = str((out / PROFILE_FILE).resolve())
        rc.profile_source = str(p.path)
        rc.profile_sha256 = p.sha256
        rc.footprint = p.footprint
        rc.footprint_radius = float(p.footprint_radius)
        rc.lidar_pose = {"xyz": s.lidar_pos, "rpy": s.lidar_rpy}
        rc.ros = dict(p.ros)
        lc = cfg.lidar
        for k in LIDAR_KEYS:
            if lc.get(k) is None:
                lc[k] = p.lidar[k]


def datagen_overrides(ref: str) -> dict[str, Any]:
    """Map-generation settings implied by a profile: inflation radius + the profile's ``datagen`` block."""
    p = load_profile(ref)
    overrides: dict[str, Any] = {"robot_radius": p.footprint_radius}
    if p.footprint is not None:
        # A rectangular footprint traverses axis-aligned passages (maze hallways, room doors) at its
        # lateral half-width, not its circumscribed radius -- the same ``passage_radius`` the benchmark
        # tight-passage scenarios use. Datagen inflates axis-aligned generators with this narrower value.
        overrides["footprint"] = [list(pt) for pt in p.footprint]
        overrides["passage_radius"] = max(abs(y) for _, y in p.footprint)
    overrides.update(p.datagen)
    overrides["robot_id"] = p.id
    return overrides
