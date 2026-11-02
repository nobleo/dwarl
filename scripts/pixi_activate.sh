# Sourced by pixi on activation: no manual `source .../setup.bash` needed.
# Overlays your robot workspace (built by `pixi run build-robot`) and then DWARL's ros/install,
# so DWARL's controller takes precedence. Set DWARL_ROBOT_WS to your robot repository.

_dwarl_root="${PIXI_PROJECT_ROOT:-$(pwd)}"

# Answers saved by `dwarl setup`; variables already set in the shell take precedence.
if [ -f "$_dwarl_root/.dwarl/setup.env" ]; then
  _saved_ws="$(. "$_dwarl_root/.dwarl/setup.env" && printf '%s' "${DWARL_ROBOT_WS:-}")"
  _saved_pkgs="$(. "$_dwarl_root/.dwarl/setup.env" && printf '%s' "${DWARL_ROBOT_PACKAGES:-}")"
  [ -n "${DWARL_ROBOT_WS:-}" ] || { [ -n "$_saved_ws" ] && export DWARL_ROBOT_WS="$_saved_ws"; }
  [ -n "${DWARL_ROBOT_PACKAGES:-}" ] || { [ -n "$_saved_pkgs" ] && export DWARL_ROBOT_PACKAGES="$_saved_pkgs"; }
fi

if [ -n "${DWARL_ROBOT_WS:-}" ] && [ -d "$DWARL_ROBOT_WS/src" ]; then
  _robot_src="$(cd "$DWARL_ROBOT_WS/src" && pwd)"
  case ":${DWARL_ROBOT_PATH:-}:" in
    *":$_robot_src:"*) ;;
    *) export DWARL_ROBOT_PATH="${DWARL_ROBOT_PATH:+$DWARL_ROBOT_PATH:}$_robot_src" ;;
  esac
fi

# Only when ROS is installed (not in the slim `train` env).
if command -v ros2 >/dev/null 2>&1; then
  for _ws in "$_dwarl_root/.pixi/robot/install" "$_dwarl_root/ros/install"; do
    if [ -f "$_ws/local_setup.sh" ]; then
      AMENT_CURRENT_PREFIX="$_ws" . "$_ws/local_setup.sh"
    fi
  done
fi

unset _dwarl_root _robot_src _ws _saved_ws _saved_pkgs
