#!/bin/bash
# Source ROS, the DWARL ROS overlay, any robot overlay, and the DWARL venv; then run the command in /work.
set -e
export HOME="${HOME:-/tmp/home}"
mkdir -p "$HOME" 2>/dev/null || true
set +u
source /opt/ros/jazzy/setup.bash
source /opt/dwarl_ws/install/setup.bash
if [ -f /opt/robots_ws/install/setup.bash ]; then source /opt/robots_ws/install/setup.bash; fi
source /opt/venv/bin/activate
export PYTHONPATH="/opt/dwarl/demo${PYTHONPATH:+:$PYTHONPATH}"
cd /work
exec "$@"
