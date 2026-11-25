"""Shared compiled differential-drive model contract and geometry inspection."""

from dataclasses import dataclass

import mujoco
import numpy as np

from dwarl.robot.profile import ProfileError


@dataclass
class ModelGeometry:
    wheel_radius: float
    wheel_separation: float
    footprint_radius: float
    lidar_offset: np.ndarray
    lidar_rotation: np.ndarray
    spawn_height: float
    mass: float
    wheel_speed_limit: float


def inspect_model(model, prefix=""):
    """Validate names, transmissions and base-frame wheel geometry for URDF and MJCF."""
    m = model
    try:
        base = m.body(prefix + "base").id
        root = m.joint(prefix + "root").id
        site = m.site(prefix + "lidar").id
        wheels = [m.body(prefix + side + "_wheel").id for side in ("left", "right")]
        joints = [m.joint(prefix + side + "_wheel").id for side in ("left", "right")]
        tires = [m.geom(prefix + side + "_tire").id for side in ("left", "right")]
        motors = [m.actuator(prefix + side + "_motor").id for side in ("left", "right")]
    except KeyError as exc:
        raise ProfileError(f"robot model naming contract: {exc}") from exc
    if m.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE or m.jnt_bodyid[root] != base:
        raise ProfileError("root must be a free joint on base")
    if m.body_rootid[base] != base or m.body_rootid[m.site_bodyid[site]] != base:
        raise ProfileError("base must be the robot root and lidar must belong to it")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rotation = d.xmat[base].reshape(3, 3)
    local = lambda pos: rotation.T @ (pos - d.xpos[base])  # noqa: E731
    left, right = [local(d.xpos[b]) for b in wheels]
    if not np.allclose(left[[0, 2]], right[[0, 2]], atol=1e-6) or left[1] <= right[1]:
        raise ProfileError("wheel bodies must be laterally aligned with left above right in base y")
    radii = []
    speeds = []
    for body, joint, tire, motor in zip(wheels, joints, tires, motors, strict=True):
        if m.body_rootid[body] != base or m.jnt_bodyid[joint] != body or m.geom_bodyid[tire] != body:
            raise ProfileError("wheel joints and tires must belong to their named wheel bodies")
        if m.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ProfileError("wheel joints must be hinges")
        if not np.allclose(rotation.T @ d.xaxis[joint], [0, 1, 0], atol=1e-6):
            raise ProfileError("wheel joint axes must point along base +y")
        if int(m.geom_type[tire]) not in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE):
            raise ProfileError("named tires must be cylinders or spheres")
        radii.append(float(m.geom_size[tire, 0]))
        if (m.actuator_trntype[motor] != mujoco.mjtTrn.mjTRN_JOINT or m.actuator_trnid[motor, 0] != joint
                or not np.allclose(m.actuator_gear[motor], [1, 0, 0, 0, 0, 0])):
            raise ProfileError("wheel motors require direct unit-gear joint transmissions")
        gain, bias = m.actuator_gainprm[motor], m.actuator_biasprm[motor]
        if (m.actuator_dyntype[motor] != mujoco.mjtDyn.mjDYN_NONE or gain[0] <= 0
                or not np.allclose(bias[:3], [0, 0, -gain[0]])):
            raise ProfileError("wheel motors must be velocity actuators")
        speeds.append(float(min(-m.actuator_ctrlrange[motor, 0], m.actuator_ctrlrange[motor, 1]))
                      if m.actuator_ctrllimited[motor] else float("inf"))
    if radii[0] <= 0 or not np.isclose(*radii):
        raise ProfileError("wheel radii must be positive and equal")
    footprint = 0.
    try:
        geom = m.geom(prefix + "chassis")
    except KeyError:
        pass
    else:
        if geom.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            footprint = float(np.linalg.norm(local(d.geom_xpos[geom.id])[:2]) + geom.size[0])
        elif geom.type[0] == mujoco.mjtGeom.mjGEOM_BOX:
            footprint = float(np.linalg.norm(local(d.geom_xpos[geom.id])[:2]) + np.linalg.norm(geom.size[:2]))
    return ModelGeometry(radii[0], float(left[1] - right[1]), footprint,
                         local(d.site_xpos[site]), rotation.T @ d.site_xmat[site].reshape(3, 3),
                         float(radii[0] - left[2]), float(m.body_subtreemass[base]), min(speeds))
