"""Robot profiles: the single, robot-owned description DWARL needs to train and deploy a policy.

A profile is a ``dwarl_robot.yaml`` that lives in the *robot's* repository (typically in a small
``<robot>_dwarl`` ROS package). It references the robot's own URDF/xacro and adds what a URDF cannot
express: drive/actuator model, command limits, footprint, the (post-filter) lidar the policy sees, extra
physics (friction, joint springs/damping, contact simplifications) and the ROS interface used to deploy
and benchmark. See ``docs/integrate-robot.md`` for the guide; ``dwarl robot check <ref>`` validates one.

Profile references (``resolve_profile``):
    path/to/dwarl_robot.yaml | path/to/dir      a file, or a directory containing one
    <ros package>                                ``share/<pkg>/dwarl_robot.yaml`` (sourced workspace) or a
                                                 source package found on ``$DWARL_ROBOT_PATH``
    <builtin>                                    ``dwarl/robots/<name>`` (e.g. ``diffbot``)
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

PROFILE_FILE = "dwarl_robot.yaml"
SCHEMA = "dwarl_robot/1"


class ProfileError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ packages
def builtin_dir() -> Path:
    return Path(str(resources.files("dwarl") / "robots"))


def _package_name(pkg_xml: Path) -> str | None:
    try:
        import xml.etree.ElementTree as ET

        el = ET.parse(pkg_xml).getroot().find("name")
        return el.text.strip() if el is not None and el.text else None
    except Exception:
        return None


def _source_roots(extra: list[Path] | None = None) -> list[Path]:
    roots = [Path(p) for p in os.environ.get("DWARL_ROBOT_PATH", "").split(os.pathsep) if p]
    return [*(extra or []), *roots]


def find_package(name: str, hints: list[Path] | None = None) -> Path | None:
    """Locate a ROS package directory: installed share (``AMENT_PREFIX_PATH``) first, then source trees on
    ``$DWARL_ROBOT_PATH`` and ``hints`` (e.g. the workspace a profile lives in). No ROS install needed."""
    for root in _source_roots(hints):
        if not root.is_dir():
            continue
        if (root / "package.xml").exists() and _package_name(root / "package.xml") == name:
            return root
        for pkg_xml in root.rglob("package.xml"):
            if any(p in ("build", "install", "log", ".git") for p in pkg_xml.parts):
                continue
            if pkg_xml.parent.name == name or _package_name(pkg_xml) == name:
                if _package_name(pkg_xml) == name:
                    return pkg_xml.parent
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if prefix and (Path(prefix) / "share" / name / "package.xml").exists():
            return Path(prefix) / "share" / name
    return None


def _workspace_hints(path: Path) -> list[Path]:
    """Directories to scan for sibling packages of a profile: the nearest ``src`` ancestor (colcon
    workspace) or else the package's parent directory."""
    for p in path.resolve().parents:
        if p.name == "src":
            return [p]
    return [path.resolve().parent.parent]


def resolve_uri(uri: str, base: Path, hints: list[Path] | None = None) -> Path:
    """``package://pkg/rel``, ``$(find pkg)/rel``, absolute or profile-relative path -> absolute path."""
    s = str(uri)
    for prefix in ("package://", "$(find "):
        if s.startswith(prefix):
            rest = s[len(prefix) :]
            if prefix == "$(find ":
                pkg, _, rel = rest.partition(")")
                rel = rel.lstrip("/")
            else:
                pkg, _, rel = rest.partition("/")
            root = find_package(pkg, hints)
            if root is None:
                raise ProfileError(f"ROS package '{pkg}' not found (source the workspace or add its src dir to "
                                   f"$DWARL_ROBOT_PATH)")  # fmt: skip
            return root / rel
    p = Path(os.path.expandvars(os.path.expanduser(s)))
    return p if p.is_absolute() else (base / p).resolve()


def resolve_profile(ref: str | Path) -> Path:
    """Profile reference -> path of its ``dwarl_robot.yaml``."""
    s = str(ref)
    p = Path(os.path.expanduser(s))
    if p.is_file():
        return p.resolve()
    if p.is_dir() and (p / PROFILE_FILE).is_file():
        return (p / PROFILE_FILE).resolve()
    if p.is_dir():
        candidates = sorted(q.resolve() for q in p.rglob(PROFILE_FILE)
                            if not any(part in {"build", "install", "log", ".git", "policies", "runs", ".venv"}
                                       for part in q.relative_to(p).parts))
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            raise ProfileError("multiple robot integrations; select an explicit profile:\n" +
                               "\n".join(str(q) for q in candidates))
        raise ProfileError(f"no {PROFILE_FILE} in {p}")
    if (builtin_dir() / s / PROFILE_FILE).is_file():
        return builtin_dir() / s / PROFILE_FILE
    if "/" not in s:
        pkg = find_package(s)
        if pkg is not None and (pkg / PROFILE_FILE).is_file():
            return (pkg / PROFILE_FILE).resolve()
    builtins = sorted(d.name for d in builtin_dir().iterdir() if (d / PROFILE_FILE).is_file())
    raise ProfileError(f"robot profile '{ref}' not found: give a path, a ROS package with {PROFILE_FILE} "
                       f"(sourced, or on $DWARL_ROBOT_PATH) or a builtin {builtins}")  # fmt: skip


# ------------------------------------------------------------------------------------------------ profile
_DEFAULT_LIDAR = dict(noise_std=0.01, dropout_prob=0.005)
_DEFAULT_ROS = dict(base_frame="base_link", odom_frame="odom", map_frame="map", scan_topic="scan",
                    cmd_vel_topic="cmd_vel", cmd_vel_stamped=True, odom_topic="odom", sim_launch=None)  # fmt: skip


@dataclass
class RobotProfile:
    """Parsed + validated ``dwarl_robot.yaml``. ``raw`` is the file content, ``path`` its location."""

    path: Path
    raw: dict[str, Any]
    id: str = ""
    limits: dict[str, float] = field(default_factory=dict)
    lidar: dict[str, Any] = field(default_factory=dict)
    footprint: list[list[float]] | None = None  # polygon (base frame) or None for a circle
    footprint_radius: float = 0.0  # circumscribed radius around the base origin
    ros: dict[str, Any] = field(default_factory=dict)
    datagen: dict[str, Any] = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return self.path.parent

    @property
    def hints(self) -> list[Path]:
        return _workspace_hints(self.path)

    def uri(self, value: str) -> Path:
        return resolve_uri(value, self.dir, self.hints)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    # policy-facing contract (what the trained network depends on)
    def training_contract(self) -> dict[str, Any]:
        return {"id": self.id, "limits": self.limits, "lidar": {k: self.lidar[k] for k in
                ("num_rays", "fov_deg", "min_range", "max_range")}}  # fmt: skip


def _req(d: dict, key: str, where: str) -> Any:
    if key not in d or d[key] is None:
        raise ProfileError(f"{where}: missing required key '{key}'")
    return d[key]


def load_profile(ref: str | Path) -> RobotProfile:
    path = resolve_profile(ref)
    from dwarl.robot.schema import parse_schema

    try:
        raw = parse_schema(yaml.safe_load(path.read_text()) or {})
    except Exception as exc:
        raise ProfileError(f"{path}: {exc}") from exc
    where = str(path)
    schema = raw.get("schema", SCHEMA)
    if schema != SCHEMA:
        raise ProfileError(f"{where}: unsupported schema '{schema}' (expected {SCHEMA})")
    rid = str(_req(raw, "id", where))
    if not rid.strip():
        raise ProfileError(f"{where}: id must not be empty")
    desc = _req(raw, "description", where)
    if not ("urdf" in desc) ^ ("mjcf" in desc):
        raise ProfileError(f"{where}: description needs exactly one of 'urdf' or 'mjcf'")
    drive = _req(raw, "drive", where)
    if drive.get("type", "diff") != "diff":
        raise ProfileError(f"{where}: only drive.type 'diff' is supported")
    if "urdf" in desc:
        for k in ("left_wheel_joint", "right_wheel_joint"):
            _req(drive, k, f"{where}: drive")
    lim = _req(raw, "limits", where)
    limits = {k: float(_req(lim, k, f"{where}: limits")) for k in ("v_min", "v_max", "w_max", "a_max", "alpha_max")}
    if not limits["v_min"] < limits["v_max"] or limits["v_max"] <= 0:
        raise ProfileError(f"{where}: limits need v_min < v_max and v_max > 0")
    lid = {**_DEFAULT_LIDAR, **_req(raw, "lidar", where)}
    for k in ("num_rays", "fov_deg", "min_range", "max_range"):
        _req(lid, k, f"{where}: lidar")
    if "urdf" in desc:
        _req(lid, "link", f"{where}: lidar")
    lid["num_rays"], lid["fov_deg"] = int(lid["num_rays"]), float(lid["fov_deg"])
    if not 0 < lid["fov_deg"] <= 360:
        raise ProfileError(f"{where}: lidar.fov_deg must be in (0, 360]")
    fp = _req(raw, "footprint", where)
    poly = None
    if "polygon" in fp:
        poly = [[float(x), float(y)] for x, y in fp["polygon"]]
        if len(poly) < 3:
            raise ProfileError(f"{where}: footprint.polygon needs >= 3 points")
        radius = max(math.hypot(x, y) for x, y in poly)
        turns = [(poly[(i+1) % len(poly)][0] - x) * (v - y)
                 - (poly[(i+1) % len(poly)][1] - y) * (u - x)
                 for i, (x, y) in enumerate(poly) for j, (u, v) in enumerate(poly)
                 if j not in (i, (i+1) % len(poly))]
        if not (all(t > 0 for t in turns) or all(t < 0 for t in turns)):
            raise ProfileError(f"{where}: footprint.polygon must be strictly convex and ordered")
    else:
        radius = float(_req(fp, "radius", f"{where}: footprint (polygon or radius)"))
    if radius <= 0 or float(fp.get("radius", radius)) < radius:
        raise ProfileError(f"{where}: footprint.radius must be positive and enclose the polygon")
    return RobotProfile(
        path=path, raw=raw, id=rid, limits=limits, lidar=lid, footprint=poly,
        footprint_radius=float(fp.get("radius", radius)), ros={**_DEFAULT_ROS, **(raw.get("ros") or {})},
        datagen=copy.deepcopy(raw.get("datagen") or {}),
    )  # fmt: skip


def dump_yaml(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False)
