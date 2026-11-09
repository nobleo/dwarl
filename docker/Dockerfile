# DWARL with ROS 2 Jazzy, Gazebo Harmonic and Nav2 (DWARL, DWB, MPPI, RPP controllers).
# Build through docker/compose.yaml; see docker/README.md.
FROM osrf/ros:jazzy-desktop AS base
ENV DEBIAN_FRONTEND=noninteractive
# Upgrade first: the base image's ROS libraries must match the Nav2 packages installed next
# (otherwise controller_server fails on a Fast-CDR symbol lookup).
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
      python3-colcon-common-extensions python3-rosdep git curl less nano \
      ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-nav2-minimal-tb3-sim \
      ros-jazzy-nav2-regulated-pure-pursuit-controller ros-jazzy-nav2-mppi-controller \
      ros-jazzy-ros-gz ros-jazzy-gz-ros2-control ros-jazzy-ros2-controllers ros-jazzy-ros2-control \
      ros-jazzy-xacro ros-jazzy-robot-state-publisher \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /usr/local/bin/
# Expose the GPU for CUDA and OpenGL when the NVIDIA container toolkit is present (compose.gpu.yaml).
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

# ---------------------------------------------------------------- DWARL Python environment
# The venv uses the system Python so that ROS's rclpy/gz bindings load inside it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON=/usr/bin/python3.12 UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
WORKDIR /opt/dwarl
COPY pyproject.toml uv.lock README.md ./
COPY runtime runtime
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --group dev --no-install-project

# ---------------------------------------------------------------- DWARL ROS packages
COPY ros/src /opt/dwarl/ros/src
RUN . /opt/ros/jazzy/setup.sh && apt-get update \
    && rosdep install --from-paths /opt/dwarl/ros/src --ignore-src -r -y \
    && rm -rf /var/lib/apt/lists/* \
    && colcon build --base-paths /opt/dwarl/ros/src --build-base /tmp/build --install-base /opt/dwarl_ws/install \
         --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    && rm -rf /tmp/build log

# ---------------------------------------------------------------- your robots (/opt/robots_ws), optional
# ROS packages from src/ of the `robots` build context (ROBOTS_WS in docker/.env; default
# docker/robots_ws, empty). Dependencies that rosdep cannot resolve are skipped.
RUN --mount=type=bind,from=robots,target=/tmp/robots_ws \
    if [ -z "$(find /tmp/robots_ws/src -name package.xml 2>/dev/null | head -n 1)" ]; then \
      echo "No robot packages in the robots context (ROBOTS_WS)"; \
    else \
      mkdir -p /opt/robots_ws && cp -r /tmp/robots_ws/src /opt/robots_ws/src \
      && . /opt/ros/jazzy/setup.sh && . /opt/dwarl_ws/install/setup.sh && apt-get update \
      && { rosdep install --from-paths /opt/robots_ws/src --ignore-src -r -y \
           || echo "rosdep: some robot dependencies were not installed"; } \
      && rm -rf /var/lib/apt/lists/* \
      && colcon build --base-paths /opt/robots_ws/src --build-base /tmp/build --install-base /opt/robots_ws/install \
      && rm -rf /tmp/build log; \
    fi

# ---------------------------------------------------------------- DWARL itself (editable: a bind mount can override it)
COPY . /opt/dwarl
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --group dev
COPY docker/entrypoint.sh /usr/local/bin/dwarl-entrypoint
RUN chmod 0755 /usr/local/bin/dwarl-entrypoint && mkdir -p /work && chmod 0777 /work
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/dwarl-entrypoint"]
CMD ["bash"]
