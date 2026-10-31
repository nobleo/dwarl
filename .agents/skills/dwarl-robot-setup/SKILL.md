---
name: dwarl-robot-setup
description: Set up a new differential-drive robot for DWARL training, testing, ONNX export and Nav2 deployment. Use when importing a robot model/profile, mapping lidar or command interfaces, checking simulator adapters, or preparing a reproducible first policy run for a user's own robot.
---

# Robot setup for DWARL

Read `AGENTS.md` and the robot integration documentation under `docs/` before editing
the integration. Use the existing profile schema, templates and commands rather than
copying robot-specific paths from a previous study.

## Inspect the contract

Locate the user's URDF/Xacro, meshes, robot integration directory and ROS launch files.
Establish differential-drive geometry, base/odom/map frames, physical footprint,
wheel radius/separation, commanded velocity and acceleration limits, control period,
lidar pose/field of view/range/history and Twist versus TwistStamped interface.

Check the full sensor transform: upside-down scanners and nonzero offsets affect ray
direction and clearance. A front scanner does not observe the rear just because a
rear collision-monitor polygon is configured. Avoid silently substituting a circular
footprint or treating an inflated costmap radius as the physical footprint.

## Bring up the smallest useful workflow

1. Use the documented `dwarl robot` inspection/check commands to verify assets and
   frames. Preserve the original model and existing policies.
2. Inspect `dwarl train <integration> --print-config` and `--dry-run`. The normal
   workflow is train → test → deploy; advanced flags remain optional.
3. Freeze the robot contract and use robot-sized data. Keep training, validation and
   test splits distinct. Check dataset metadata rather than reusing another robot's
   geometry because it happens to load.
4. Start with a bounded training budget. If enabling validation/early stopping, use
   intervals that are multiples of save interval and record patience/minimum gain.
5. Test an exact checkpoint on a small balanced MuJoCo validation set. Inspect
   individual trajectories and motion traces, not only the average success rate.
6. Export a bundle with its ONNX model, deployment contract, robot identity and
   checkpoint hash. Use runtime parity checks and recorded-input model replay where
   available; desktop replay timing is not the robot CPU measurement.
7. Source ROS Jazzy, the robot workspace and `ros/install/setup.bash`. Verify the
   Gazebo adapter accepts world/spawn arguments and supplies clock, scan, odometry,
   TF and the correct velocity interface. Match the simulator entity to the profile's
   `ros.gazebo_model` (default `robot`); do not hard-code that default for every robot.
   Run one smoke case before broader tests.
8. Check actual clearance, commands, measured motion, safety-monitor state and
   localization differences. A successful launch is not successful navigation.

Use `.agents/skills/dwarl-performance-loop/SKILL.md` for subsequent report-driven
tuning. Keep robot-neutral public instructions, record the exact commands and inputs,
and distinguish simulator evidence from physical-robot validation.
