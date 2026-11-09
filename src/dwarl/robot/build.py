"""Profile -> MuJoCo model (the simulator's robot), following the "URDF + sidecar physics" approach.

1. ``description.urdf`` is expanded with xacro (ROS' ``xacro`` python module; no sourced workspace needed:
   ``package://`` / ``$(find pkg)`` are resolved by :func:`dwarl.robot.profile.find_package`).
2. Contact geometry is simplified for stable, fast training: only the drive wheels keep their URDF
   collision shapes; the chassis becomes an extruded footprint prism and casters become frictionless
   support spheres (``collision`` section). Visuals are dropped (``collision.keep_visuals`` keeps them).
3. MuJoCo parses the URDF into an ``MjSpec``; we add the free joint, wheel velocity actuators, a ``lidar``
   site on the lidar link and apply ``physics`` overrides (per-link friction, per-joint stiffness/damping/
   armature). Names are normalised to the simulator contract
   (``base``/``root``/``left_wheel``/``right_wheel``/``left_motor``/``right_motor``/``left_tire``/
   ``right_tire``/``lidar``), see :mod:`dwarl.simulation.world`.

``description.mjcf`` profiles (builtin toy robots) must already follow that contract and are used as is.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from dwarl.robot.profile import ProfileError, RobotProfile, find_package

ROBOT_GROUP = 2
_XACRO_LOCK = threading.RLock()


@contextmanager
def _package_resolver(xacro, profile):
    """Serialize and restore xacro's process-global package resolver, even on failure."""
    with _XACRO_LOCK:
        try:
            from ament_index_python import packages
        except ImportError:
            yield
            return
        targets = [packages]
        substitution = getattr(xacro, "substitution_args", None)
        if substitution is not None and hasattr(substitution, "get_package_share_directory"):
            targets.append(substitution)
        originals = [(target, target.get_package_share_directory) for target in targets]
        original = packages.get_package_share_directory

        def share(name, *args, **kwargs):
            try:
                return original(name, *args, **kwargs)
            except LookupError:
                path = find_package(name, profile.hints)
                if path is None:
                    raise
                return str(path)

        try:
            for target, _ in originals:
                target.get_package_share_directory = share
            yield
        finally:
            for target, original in originals:
                target.get_package_share_directory = original


# ------------------------------------------------------------------------------------------------ xacro
def _import_xacro():
    try:
        import xacro  # type: ignore

        return xacro
    except ImportError:
        pass
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for root in [*os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep), *sorted(Path("/opt/ros").glob("*"))]:
        site = Path(root) / "lib" / py / "site-packages"
        if (site / "xacro").is_dir():
            sys.path.append(str(site))
            import xacro  # type: ignore

            return xacro
    raise ProfileError("xacro not found: install ROS' xacro (apt ros-<distro>-xacro) or `pip install xacro`, "
                       "or point description.urdf at a plain .urdf")  # fmt: skip


def expand_urdf(profile: RobotProfile) -> str:
    """URDF text of the profile's robot (xacro expanded with ``description.xacro_args``)."""
    desc = profile.raw["description"]
    path = profile.uri(desc["urdf"])
    if not path.is_file():
        raise ProfileError(f"URDF not found: {path}")
    if path.suffix != ".xacro" and not desc.get("xacro_args"):
        return path.read_text()
    xacro = _import_xacro()
    mappings = {str(k): str(v) for k, v in (desc.get("xacro_args") or {}).items()}
    with _package_resolver(xacro, profile):
        doc = xacro.process_file(str(path), mappings=mappings)
    return doc.toxml()


# ------------------------------------------------------------------------------------------------ urdf utils
def _rpy_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])  # fmt: skip


def _origin(el: ET.Element | None) -> np.ndarray:
    T = np.eye(4)
    if el is None:
        return T
    xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
    T[:3, :3], T[:3, 3] = _rpy_matrix(rpy), xyz
    return T


def link_transform(root: ET.Element, base: str, link: str) -> np.ndarray:
    """4x4 pose of ``link`` in ``base`` from the URDF joint origins (zero joint positions)."""
    parent_of = {j.find("child").get("link"): j for j in root.findall("joint")}  # type: ignore[union-attr]
    T, cur, visited = np.eye(4), link, set()
    while cur != base:
        if cur in visited:
            raise ProfileError(f"cycle in URDF ancestors of '{link}'")
        visited.add(cur)
        j = parent_of.get(cur)
        if j is None:
            raise ProfileError(f"link '{link}' is not a descendant of base link '{base}'")
        T = _origin(j.find("origin")) @ T
        cur = j.find("parent").get("link")  # type: ignore[union-attr]
    return T


def _subtree(root: ET.Element, link: str) -> set[str]:
    children: dict[str, list[str]] = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j.find("child").get("link"))  # type: ignore
    out, stack = set(), [link]
    while stack:
        cur = stack.pop()
        if cur in out:
            raise ProfileError("cycle or duplicate child in URDF")
        out.add(cur)
        stack += children.get(cur, [])
    return out


def _joint(root: ET.Element, name: str) -> ET.Element:
    for j in root.findall("joint"):
        if j.get("name") == name:
            return j
    raise ProfileError(f"joint '{name}' not found in the URDF")


def _wheel_radius(link_el: ET.Element | None) -> float | None:
    if link_el is None:
        return None
    for c in link_el.findall("collision"):
        g = c.find("geometry")
        for kind in ("cylinder", "sphere"):
            e = g.find(kind) if g is not None else None
            if e is not None:
                return float(e.get("radius"))
    return None


@dataclass
class UrdfInfo:
    base_link: str
    left_link: str
    right_link: str
    wheel_radius: float
    wheel_separation: float
    wheel_z: float  # wheel axle height in the base frame
    lidar_pose: np.ndarray  # 4x4 lidar link pose in the base frame


def prepare_urdf(profile: RobotProfile, urdf: str) -> tuple[str, UrdfInfo]:
    """Simplify the URDF for MuJoCo and extract the drive/lidar geometry."""
    root = ET.fromstring(urdf)
    desc, drive, col = profile.raw["description"], profile.raw["drive"], profile.raw.get("collision") or {}
    base = desc.get("base_link", "base_link")
    links = {lk.get("name"): lk for lk in root.findall("link")}
    if base not in links:
        raise ProfileError(f"base link '{base}' not in URDF (links: {sorted(links)})")
    jl, jr = _joint(root, drive["left_wheel_joint"]), _joint(root, drive["right_wheel_joint"])
    ll, rl = jl.find("child").get("link"), jr.find("child").get("link")  # type: ignore[union-attr]
    for j in (jl, jr):
        if j.get("type") not in ("continuous", "revolute"):
            raise ProfileError(f"wheel joint '{j.get('name')}' must be continuous/revolute")
    Tl, Tr = link_transform(root, base, ll), link_transform(root, base, rl)
    if ll == rl or not np.allclose(Tl[[0, 2], 3], Tr[[0, 2], 3], atol=1e-6) or Tl[1, 3] <= Tr[1, 3]:
        raise ProfileError("drive wheels must be distinct and aligned laterally, left at positive relative y")
    for joint, transform in ((jl, Tl), (jr, Tr)):
        axis_el = joint.find("axis")
        axis = np.array([float(v) for v in (axis_el.get("xyz") if axis_el is not None else "1 0 0").split()])
        if not np.allclose(transform[:3, :3] @ axis, [0, 1, 0], atol=1e-6):
            raise ProfileError("drive wheel axes must point along base +y")
    left_radius, right_radius = _wheel_radius(links.get(ll)), _wheel_radius(links.get(rl))
    if left_radius and right_radius and not np.isclose(left_radius, right_radius):
        raise ProfileError("drive wheels must have matching radii")
    radius = float(drive.get("wheel_radius") or _wheel_radius(links.get(ll)) or 0.0)
    if radius <= 0:
        raise ProfileError("wheel radius: no cylinder/sphere collision on the wheel link; set drive.wheel_radius")
    if any(r is not None and not np.isclose(radius, r) for r in (left_radius, right_radius)):
        raise ProfileError("drive.wheel_radius must agree with wheel collision geometry")
    lidar_link = profile.lidar["link"]
    if lidar_link not in links:
        raise ProfileError(f"lidar link '{lidar_link}' not in URDF")
    info = UrdfInfo(base_link=base, left_link=ll, right_link=rl, wheel_radius=radius,
                    wheel_separation=float(np.linalg.norm(Tl[:3, 3] - Tr[:3, 3])),
                    wheel_z=float(Tl[2, 3]), lidar_pose=link_transform(root, base, lidar_link))  # fmt: skip

    # --- contact simplification
    remove: set[str] = set()
    for lk in col.get("remove_links") or []:
        if lk not in links:
            raise ProfileError(f"collision.remove_links: '{lk}' not in URDF")
        if lidar_link in _subtree(root, lk) or ll in _subtree(root, lk) or rl in _subtree(root, lk):
            raise ProfileError(f"collision.remove_links: '{lk}' would remove a wheel or the lidar")
        remove |= _subtree(root, lk)
    keep = set(col.get("keep_collision_links") or []) | {ll, rl}
    if keep - links.keys():
        raise ProfileError(f"collision.keep_collision_links not in URDF: {sorted(keep - links.keys())}")
    keep_visuals = bool(col.get("keep_visuals", False))
    for name, lk in links.items():
        if name in remove:
            root.remove(lk)
            continue
        for tag in ("collision",) if keep_visuals else ("collision", "visual"):
            if tag == "collision" and name in keep:
                continue
            for el in lk.findall(tag):
                lk.remove(el)
    for j in list(root.findall("joint")):
        if j.find("child").get("link") in remove:  # type: ignore[union-attr]
            root.remove(j)
    for m in root.iter("mesh"):
        m.set("filename", str(profile.uri(m.get("filename", ""))))
    for t in root.findall("gazebo") + root.findall("ros2_control") + root.findall("transmission"):
        root.remove(t)
    mj = ET.SubElement(root, "mujoco")
    ET.SubElement(mj, "compiler", fusestatic="false", discardvisual="false" if keep_visuals else "true",
                  strippath="false", balanceinertia="true")  # fmt: skip
    return ET.tostring(root, encoding="unicode"), info


# ------------------------------------------------------------------------------------------------ build
def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1).astype(np.float64))
    return q


def _by_name(items, name: str):
    for it in items:
        if it.name == name:
            return it
    raise ProfileError(f"'{name}' not found in the model")


def _prism_mesh(polygon: list[list[float]], z0: float, z1: float) -> list[float]:
    verts = []
    for z in (z0, z1):
        for x, y in polygon:
            verts += [x, y, z]
    return verts


def build_spec(profile: RobotProfile) -> mujoco.MjSpec:
    """MuJoCo spec of the robot following the simulator naming contract."""
    desc = profile.raw["description"]
    if "mjcf" in desc:
        return mujoco.MjSpec.from_file(str(profile.uri(desc["mjcf"])))
    urdf, info = prepare_urdf(profile, expand_urdf(profile))
    spec = mujoco.MjSpec.from_string(urdf)
    spec.modelname = profile.id
    drive, col, phys = profile.raw["drive"], profile.raw.get("collision") or {}, profile.raw.get("physics") or {}

    base = _by_name(spec.bodies, info.base_link)
    lidar_body = _by_name(spec.bodies, profile.lidar["link"])
    base.name = "base"
    base.add_freejoint(name="root")
    for side, link, jname in (("left", info.left_link, drive["left_wheel_joint"]),
                              ("right", info.right_link, drive["right_wheel_joint"])):  # fmt: skip
        body = _by_name(spec.bodies, link)
        body.name = f"{side}_wheel"
        _by_name(spec.joints, jname).name = f"{side}_wheel"
        tires = [g for g in spec.geoms if g.parent.name == body.name and g.type in
                 (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE)]  # fmt: skip
        if tires:
            tires[0].name = f"{side}_tire"
        else:  # no URDF collision on the wheel: add one from the profile radius
            body.add_geom(name=f"{side}_tire", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[info.wheel_radius, 0.02, 0],
                          quat=[math.sqrt(0.5), math.sqrt(0.5), 0, 0])  # fmt: skip
    # lidar site on the lidar link (orientation kept: deployments use the full mounting transform)
    lidar_body.add_site(name="lidar", size=[0.01, 0, 0], rgba=[1, 0, 0, 1])

    # chassis: extruded footprint prism (or a cylinder for a circular footprint)
    z0, z1 = (float(v) for v in col.get("chassis_z", [info.wheel_z - info.wheel_radius + 0.05, 0.4]))
    mass = float(col.get("chassis_mass", 0.0))
    # own welded body: URDF bodies carry explicit inertials, which would ignore the geom's mass
    holder = base.add_body(name="chassis") if mass > 0 else base
    if mass > 0:  # siblings of the wheels: exclude, the prism encloses them in the footprint
        for side in ("left", "right"):
            spec.add_exclude(bodyname1="chassis", bodyname2=f"{side}_wheel")
    if profile.footprint is not None:
        spec.add_mesh(name="chassis_prism", uservert=_prism_mesh(profile.footprint, z0, z1))
        g = holder.add_geom(name="chassis", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="chassis_prism")
    else:
        r = profile.footprint_radius
        g = holder.add_geom(name="chassis", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, (z1 - z0) / 2, 0],
                          pos=[0, 0, (z0 + z1) / 2])  # fmt: skip
    g.rgba = [0.45, 0.45, 0.5, 0.6]
    if mass > 0:
        g.mass = mass
    else:
        g.density = 0.0  # pure contact shape; the URDF inertials carry the mass
    for k, s in enumerate(col.get("supports") or []):
        base.add_geom(name=f"support{k}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[float(s["radius"]), 0, 0],
                      pos=[float(v) for v in s["pos"]], condim=1, friction=[0, 0, 0], density=0.0, priority=1,
                      rgba=[0.2, 0.2, 0.2, 1])  # fmt: skip

    # contacts: every robot geom collides with the world, groups for the viewer / lidar exclusion
    for geom in spec.geoms:
        geom.group = ROBOT_GROUP
        if geom.contype or geom.conaffinity:
            geom.contype, geom.conaffinity = 1, 1
    default_friction = phys.get("friction")
    for geom in spec.geoms:
        if default_friction is not None and not geom.name.startswith("support"):
            geom.friction = [float(v) for v in default_friction]
    for f in phys.get("link_friction") or []:
        names = {_body_name(f["link"], info, profile)}
        _by_name(spec.bodies, next(iter(names)))
        for geom in spec.geoms:
            if geom.parent.name in names:
                geom.friction = [float(v) for v in f["friction"]]
    for jc in phys.get("joints") or []:
        name = jc["joint"]
        name = {drive["left_wheel_joint"]: "left_wheel", drive["right_wheel_joint"]: "right_wheel"}.get(name, name)
        j = _by_name(spec.joints, name)
        for k in ("stiffness", "damping", "armature", "frictionloss"):
            if k in jc:
                cur = getattr(j, k)
                if np.ndim(cur):  # MuJoCo >= 3.4: polynomial coefficients, [0] is the linear term
                    vec = np.zeros_like(cur)
                    vec[0] = float(jc[k])
                    setattr(j, k, vec)
                else:
                    setattr(j, k, float(jc[k]))
        if "reference" in jc:
            j.springref = float(jc["reference"])

    act = {"kv": 1.0, "max_wheel_speed": 30.0, "max_torque": 5.0, **(drive.get("actuator") or {})}
    for side in ("left", "right"):
        a = spec.add_actuator(name=f"{side}_motor", target=f"{side}_wheel", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_velocity(kv=float(act["kv"]))
        a.ctrlrange = [-float(act["max_wheel_speed"]), float(act["max_wheel_speed"])]
        a.ctrllimited = 1
        a.forcerange = [-float(act["max_torque"]), float(act["max_torque"])]
        a.forcelimited = 1
    return spec


def _body_name(link: str, info: UrdfInfo, profile: RobotProfile) -> str:
    return {info.base_link: "base", info.left_link: "left_wheel", info.right_link: "right_wheel"}.get(link, link)


# ------------------------------------------------------------------------------------------------ inspection
@dataclass
class RobotSummary:
    id: str
    mass: float
    wheel_radius: float
    wheel_separation: float
    footprint_radius: float
    lidar_pos: list[float]  # lidar site position in the base frame
    lidar_rpy: list[float]
    lidar_upside_down: bool
    spawn_height: float
    nq: int
    ngeom: int
    wheel_speed_limit: float

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def summarize(profile: RobotProfile, xml: str) -> RobotSummary:
    from dwarl.robot.model import inspect_model

    m = mujoco.MjModel.from_xml_string(xml)
    geometry = inspect_model(m)
    pos, R = geometry.lidar_offset, geometry.lidar_rotation
    rpy = [math.atan2(R[2, 1], R[2, 2]), -math.asin(max(-1.0, min(1.0, R[2, 0]))), math.atan2(R[1, 0], R[0, 0])]
    return RobotSummary(
        id=profile.id, mass=geometry.mass, wheel_radius=geometry.wheel_radius,
        wheel_separation=geometry.wheel_separation, footprint_radius=profile.footprint_radius,
        lidar_pos=[float(v) for v in pos], lidar_rpy=[float(v) for v in rpy], lidar_upside_down=bool(R[2, 2] < 0),
        spawn_height=geometry.spawn_height, nq=int(m.nq), ngeom=int(m.ngeom),
        wheel_speed_limit=geometry.wheel_speed_limit,
    )  # fmt: skip


def build_xml(profile: RobotProfile) -> str:
    spec = build_spec(profile)
    from dwarl.robot.model import inspect_model

    inspect_model(spec.compile())
    root = ET.fromstring(spec.to_xml())
    compiler = root.find("compiler")
    desc = profile.raw["description"]
    source_dir = profile.uri(desc.get("mjcf") or desc["urdf"]).parent
    for asset in root.findall("./asset/*"):
        filename = asset.get("file")
        if filename:
            key = "meshdir" if asset.tag == "mesh" else "texturedir"
            directory = compiler.get(key, "") if compiler is not None else ""
            asset.set("file", str((source_dir / directory / filename).resolve()))
    if compiler is not None:
        for key in ("meshdir", "texturedir", "assetdir"):
            compiler.attrib.pop(key, None)
    return ET.tostring(root, encoding="unicode")


def check_against_urdf(profile: RobotProfile, summary: RobotSummary) -> list[str]:
    """Consistency warnings between profile and model (things that silently break a deployment)."""
    warn = []
    lim = profile.limits
    if math.isfinite(summary.wheel_speed_limit):
        wmax_wheels = summary.wheel_speed_limit * summary.wheel_radius
        need = max(abs(lim["v_min"]), lim["v_max"]) + lim["w_max"] * summary.wheel_separation / 2
        if need > wmax_wheels + 1e-6:
            warn.append(f"actuator max_wheel_speed allows {wmax_wheels:.2f} m/s at the rim, limits need {need:.2f}")
    if profile.footprint is not None:
        lx, ly = summary.lidar_pos[:2]
        if not _inside(profile.footprint, lx, ly):
            warn.append("lidar origin lies outside the footprint polygon")
    if lim["v_min"] < 0 and profile.lidar["fov_deg"] < 300:
        warn.append(f"reversing allowed (v_min={lim['v_min']}) with a {profile.lidar['fov_deg']:.0f} deg lidar: "
                    f"the policy is blind behind the robot")  # fmt: skip
    return warn


def _inside(poly, x, y) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside
