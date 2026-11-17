# One dependency stack: the same locked Pixi default environment used on the host.
# Build through docker/compose.yaml; see docker/README.md.
FROM ghcr.io/prefix-dev/pixi:0.83.0

ENV DEBIAN_FRONTEND=noninteractive
# OS utilities and display drivers only. Python, ROS, Gazebo and build tools come from pixi.lock.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git less nano libgl1-mesa-dri libegl1 libgl1 libxkbcommon-x11-0 libxcb-cursor0 \
    && rm -rf /var/lib/apt/lists/*
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
ENV DWARL_PREBUILT_ROS=1 DWARL_ROBOT_WS=/opt/robots_ws
# Bound compilation parallelism (also useful when training is running on the host).
ARG BUILD_JOBS=2
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}

WORKDIR /opt/dwarl
COPY . .
# --locked refuses manifest/lockfile drift; it never performs a fresh dependency solve.
RUN --mount=type=cache,target=/root/.cache/rattler \
    pixi install --locked --environment default
RUN pixi run --locked build

# Optional robot sources use the same compiler, ROS underlay and build task as local development.
# Robot dependencies must be declared in pyproject.toml and locked, or supplied as source packages.
RUN --mount=type=bind,from=robots,target=/tmp/robots_ws \
    mkdir -p /opt/robots_ws/src \
    && if [ -n "$(find /tmp/robots_ws/src -name package.xml -print -quit)" ]; then \
      cp -r /tmp/robots_ws/src/. /opt/robots_ws/src/ \
      && pixi run --locked build-robot; \
    else \
      echo "No robot packages in the robots context (ROBOTS_WS)"; \
    fi

# Bake activation after both overlays are built. Startup needs no solve, download or writable prefix.
RUN pixi shell-hook --as-is --environment default --shell bash --no-completions --change-ps1 false \
      > /opt/dwarl-activate.sh
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/dwarl-entrypoint
RUN mkdir -p /work && chmod 0777 /work
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/dwarl-entrypoint"]
CMD ["bash"]
