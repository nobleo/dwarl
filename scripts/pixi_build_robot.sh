#!/usr/bin/env bash
# Builds your robot workspace (DWARL_ROBOT_WS) against pixi's ROS into .pixi/robot/, leaving the robot
# repository's own build/ and install/ untouched. DWARL_ROBOT_PACKAGES limits it to the packages DWARL
# needs (e.g. the integration package with the Gazebo adapter) and their dependencies.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "${DWARL_ROBOT_WS:-}" ]; then
  echo "No robot workspace configured (run \`pixi run setup\`); skipping."
  exit 0
fi
ws="$(cd "$DWARL_ROBOT_WS" && pwd)"
[ -d "$ws/src" ] || { echo "No src/ in $ws" >&2; exit 1; }

out="$root/.pixi/robot"
mkdir -p "$out"
args=(--base-paths "$ws/src" --build-base "$out/build" --install-base "$out/install" --symlink-install)
if [ -n "${DWARL_ROBOT_PACKAGES:-}" ]; then
  read -r -a pkgs <<< "$DWARL_ROBOT_PACKAGES"
  args+=(--packages-up-to "${pkgs[@]}")
fi
args+=(--cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo)
cd "$out"  # not $ws: a robot colcon_defaults.yaml may target hardware-only packages
colcon build "${args[@]}"
echo "Robot workspace built into $out/install (activated automatically by pixi)."
