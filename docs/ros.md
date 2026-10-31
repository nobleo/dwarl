# Native ROS setup

For the complete model → training → testing → export → Nav2 workflow, follow
[Your robot: from model to Nav2](integrate-robot.md).

Target platform: Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic. The training environment
and the lightweight robot runtime are separate packages. Loading different policy
weights must not require a ROS rebuild.

Install ROS 2 Jazzy using its Ubuntu deb instructions, then install the native dependencies:

```sh
sudo apt install python3-colcon-common-extensions python3-rosdep \
  ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-ros-gz \
  ros-jazzy-xacro ros-jazzy-robot-state-publisher
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths ros/src --ignore-src -r -y
colcon build --symlink-install --base-paths ros/src \
  --build-base ros/build --install-base ros/install
source ros/install/setup.bash
```

The controller build downloads ONNX Runtime C++ 1.20.1. Build the robot repository
and its dependencies with `--symlink-install`, following that repository's setup
instructions, and source its install workspace. Source DWARL's `ros/install/setup.bash`
afterward so its controller takes precedence. The robot's spawn adapter and referenced
mesh packages must be installed and discoverable.

Activate DWARL's Python environment after sourcing ROS:

```sh
source .venv/bin/activate
dwarl doctor --sim gazebo
dwarl test --sim both --case doorway
dwarl deploy /path/to/robot-repository
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
