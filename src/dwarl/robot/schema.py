"""Typed robot sidecar schema and physical-domain validation."""

import math
from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING, OmegaConf


@dataclass
class DescriptionCfg:
    urdf: str | None = None
    mjcf: str | None = None
    base_link: str = "base_link"
    xacro_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActuatorCfg:
    kv: float = 1.0
    max_wheel_speed: float = 30.0
    max_torque: float = 5.0


@dataclass
class DriveCfg:
    type: str = "diff"
    left_wheel_joint: str | None = None
    right_wheel_joint: str | None = None
    wheel_radius: float | None = None
    actuator: ActuatorCfg = field(default_factory=ActuatorCfg)


@dataclass
class LimitsCfg:
    v_min: float = MISSING
    v_max: float = MISSING
    w_max: float = MISSING
    a_max: float = MISSING
    alpha_max: float = MISSING


@dataclass
class LidarCfg:
    link: str | None = None
    num_rays: int = MISSING
    fov_deg: float = MISSING
    min_range: float = MISSING
    max_range: float = MISSING
    noise_std: float = .01
    dropout_prob: float = .005


@dataclass
class FootprintCfg:
    polygon: list[list[float]] | None = None
    radius: float | None = None


@dataclass
class SupportCfg:
    pos: list[float] = MISSING
    radius: float = MISSING


@dataclass
class CollisionCfg:
    remove_links: list[str] = field(default_factory=list)
    keep_collision_links: list[str] = field(default_factory=list)
    keep_visuals: bool = False
    chassis_z: list[float] | None = None
    chassis_mass: float = 0.
    supports: list[SupportCfg] = field(default_factory=list)


@dataclass
class JointCfg:
    joint: str = MISSING
    stiffness: float | None = None
    damping: float | None = None
    armature: float | None = None
    frictionloss: float | None = None
    reference: float | None = None


@dataclass
class LinkFrictionCfg:
    link: str = MISSING
    friction: list[float] = MISSING


@dataclass
class PhysicsCfg:
    friction: list[float] | None = None
    link_friction: list[LinkFrictionCfg] = field(default_factory=list)
    joints: list[JointCfg] = field(default_factory=list)


@dataclass
class LaunchCfg:
    package: str = MISSING
    file: str = MISSING


@dataclass
class RosCfg:
    base_frame: str = "base_link"
    odom_frame: str = "odom"
    map_frame: str = "map"
    scan_topic: str = "scan"
    cmd_vel_topic: str = "cmd_vel"
    cmd_vel_stamped: bool = True
    odom_topic: str = "odom"
    sim_launch: LaunchCfg | None = None


@dataclass
class DeploymentCfg:
    policy_directory: str = "policies"
    selected_policy: str = "selected_policy.yaml"
    launch: LaunchCfg | None = None


@dataclass
class ProfileCfg:
    schema: str = "dwarl_robot/1"
    id: str = MISSING
    description: DescriptionCfg = field(default_factory=DescriptionCfg)
    drive: DriveCfg = field(default_factory=DriveCfg)
    limits: LimitsCfg = field(default_factory=LimitsCfg)
    lidar: LidarCfg = field(default_factory=LidarCfg)
    footprint: FootprintCfg = field(default_factory=FootprintCfg)
    collision: CollisionCfg = field(default_factory=CollisionCfg)
    physics: PhysicsCfg = field(default_factory=PhysicsCfg)
    ros: RosCfg = field(default_factory=RosCfg)
    datagen: dict[str, Any] = field(default_factory=dict)
    deployment: DeploymentCfg = field(default_factory=DeploymentCfg)


def validate_measurements(limits, lidar):
    for key, value in limits.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"limits.{key} must be finite")
    if not limits["v_min"] < limits["v_max"] or limits["v_max"] <= 0:
        raise ValueError("limits require v_min < v_max and v_max > 0")
    for key in ("w_max", "a_max", "alpha_max"):
        if limits[key] <= 0:
            raise ValueError(f"limits.{key} must be positive")
    for key in ("num_rays", "fov_deg", "min_range", "max_range", "noise_std", "dropout_prob"):
        if not math.isfinite(float(lidar[key])):
            raise ValueError(f"lidar.{key} must be finite")
    if int(lidar["num_rays"]) != lidar["num_rays"] or lidar["num_rays"] <= 0:
        raise ValueError("lidar.num_rays must be a positive integer")
    if not 0 < lidar["fov_deg"] <= 360:
        raise ValueError("lidar.fov_deg must be in (0, 360]")
    if not 0 <= lidar["min_range"] < lidar["max_range"]:
        raise ValueError("lidar ranges require 0 <= min_range < max_range")
    if lidar["noise_std"] < 0 or not 0 <= lidar["dropout_prob"] <= 1:
        raise ValueError("invalid lidar noise_std or dropout_prob")


def parse_schema(raw):
    cfg = OmegaConf.merge(OmegaConf.structured(ProfileCfg), raw)
    data = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # Optional unset fields must not masquerade as supplied URDF/MJCF options.
    def compact(value):
        if isinstance(value, dict):
            return {k: compact(v) for k, v in value.items() if v is not None}
        if isinstance(value, list):
            return [compact(v) for v in value]
        return value
    data = compact(data)
    validate_measurements(data["limits"], data["lidar"])
    def finite(value):
        if isinstance(value, dict):
            for v in value.values():
                finite(v)
        elif isinstance(value, list):
            for v in value:
                finite(v)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("profile numbers must be finite")
    finite(data)
    for key, value in data["drive"]["actuator"].items():
        if value <= 0:
            raise ValueError(f"drive.actuator.{key} must be positive")
    if data["drive"].get("wheel_radius", 1) <= 0:
        raise ValueError("drive.wheel_radius must be positive")
    col = data["collision"]
    if col["chassis_mass"] < 0:
        raise ValueError("collision.chassis_mass must be nonnegative")
    if "chassis_z" in col and (len(col["chassis_z"]) != 2 or col["chassis_z"][0] >= col["chassis_z"][1]):
        raise ValueError("collision.chassis_z requires increasing bounds")
    for support in col["supports"]:
        if len(support["pos"]) != 3 or support["radius"] <= 0:
            raise ValueError("collision.supports require xyz and positive radius")
    phys = data["physics"]
    for friction in ([phys["friction"]] if "friction" in phys else []) + [f["friction"] for f in phys["link_friction"]]:
        if len(friction) != 3 or min(friction) < 0:
            raise ValueError("physics friction requires three nonnegative coefficients")
    for joint in phys["joints"]:
        if any(v < 0 for k, v in joint.items() if k not in ("joint", "reference")):
            raise ValueError("physics joint coefficients must be nonnegative")
    return data
