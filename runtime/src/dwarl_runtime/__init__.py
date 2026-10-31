"""DWARL policy runtime: run an exported policy bundle without ROS, torch or MuJoCo.

Depends only on numpy, pyyaml and onnxruntime. Mirrors the C++ ``dwarl_policy_core`` (and through the
golden-data tests, the training environment) so it can be embedded in any robot software stack::

    from dwarl_runtime import Policy, Scan

    policy = Policy("path/to/bundle")                  # dir with deploy.yaml + policy.onnx
    policy.set_plan(path_xy)                            # (N, 2) global path in a fixed frame (e.g. map)
    cmd = policy.step(scan, laser_pose, robot_pose, velocity)
    send(cmd.v, cmd.w)                                  # every policy.dt seconds

``laser_pose`` is the scan frame in the robot base frame: a 4x4 homogeneous matrix or ``(xyz, quat_xyzw)``.
The full 3D rotation is used, so upside-down / tilted mounts are handled.
"""

from dwarl_runtime.core import (
    Command,
    DeploySpec,
    PathTracker,
    Policy,
    Scan,
    command_from_action,
    laser_matrix,
    lidar_observation,
    state_observation,
)

__all__ = [
    "Command",
    "DeploySpec",
    "PathTracker",
    "Policy",
    "Scan",
    "command_from_action",
    "laser_matrix",
    "lidar_observation",
    "state_observation",
]
