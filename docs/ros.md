# Native ROS setup

For the complete model → training → testing → export → Nav2 workflow, follow
[Your robot: from model to Nav2](integrate-robot.md).

Target platform: Linux x86-64 with ROS 2 Jazzy and Gazebo Harmonic, installed by
[pixi](https://pixi.sh) from [RoboStack](https://robostack.github.io): no system ROS install,
`apt`, `rosdep` or `source .../setup.bash` is needed. The training environment and the
lightweight robot runtime are separate packages. Loading different policy weights must
not require a ROS rebuild.

Run the guided setup once:

```sh
pixi run setup
```

It offers the example robot, a new integration or a saved integration. Workspace answers
are stored in `.dwarl/setup.env` and active selections in `.dwarl/context.json`.
Gazebo integrations build DWARL's `ros/` packages and the robot workspace against
pixi's ROS (into `.pixi/robot/`) and offer a live check. Rerun it to change the answers. For scripts and CI:
`dwarl setup --robot-ws /path/to/robot --packages my_robot_dwarl --yes`.
`DWARL_ROBOT_WS` / `DWARL_ROBOT_PACKAGES` in the environment override the saved answers.

The robot build leaves the robot repository's own `build/` and `install/` untouched, and building
only the integration package(s) keeps hardware-only packages (drivers, CAN stacks) out. DWARL loads
the robot overlay and then its own `ros/install` (its controller takes precedence) for every Gazebo
run, also right after a build in the same command. Robot runtime dependencies missing from the `ros`
feature in `pyproject.toml` can be added with `pixi add --feature ros ros-jazzy-<package>`.
Run pixi from a terminal without another ROS workspace sourced.

Gazebo testing and deployment with a launch adapter run incremental builds first.
The same preparation runs from task shortcuts and direct CLI commands inside `pixi shell`.
MuJoCo testing does not trigger a ROS build. Use `pixi run build-robot` for an explicit rebuild.
`dwarl doctor --sim gazebo` (also part of `dwarl test --sim gazebo`) warns about Gazebo servers left
behind by an interrupted DWARL test (they share ROS domains and CPU) and a `powersave` CPU governor.

```sh
pixi run test --sim both --case doorway
pixi run deploy /path/to/robot-repository
```

The robot's planner launch needs one initial integration-package build.
Subsequent deployments update the source integration's selected-policy file and bundles;
the symlink-installed launch reads that source location, so new weights need no rebuild.

Fresh-machine installation and broader Gazebo scoring validation remain experimental.
`dwarl doctor --sim gazebo` reports missing prerequisites.

The initial localization mode is simulation odometry with an explicit spawn-frame
transform, **not AMCL validation**. Ground-truth scoring must be checked independently
of the policy's odometry and scan inputs.

Ground truth uses Jazzy's Gazebo Transport 13 and Messages 10 Python bindings directly:
the generic ROS Pose_V bridge does not preserve scene-pose model names. Each managed
test owns a ROS domain lease and Gazebo partition, and starts a fresh stack per case.
