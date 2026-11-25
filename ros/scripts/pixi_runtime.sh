#!/usr/bin/env bash
# Internal task activation after an in-command build. Users only invoke pixi.
set -euo pipefail
root="${PIXI_PROJECT_ROOT:?Run this through pixi}"
export AMENT_CURRENT_PREFIX="$root/ros/install-runtime"
# ROS setup scripts are not nounset-clean.
set +u
. "$AMENT_CURRENT_PREFIX/local_setup.sh"
set -u
exec ros2 launch dwarl_bringup navigation.launch.py "$@"
