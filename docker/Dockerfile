FROM ghcr.io/prefix-dev/pixi:0.83.0

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git libgl1-mesa-dri libegl1 libgl1 libxkbcommon-x11-0 libxcb-cursor0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/dwarl
COPY pixi.lock pixi.toml ./
COPY dwarl/pyproject.toml dwarl/README.md dwarl/LICENSE dwarl/
RUN --mount=type=cache,target=/root/.cache/rattler \
    pixi install --locked --skip dwarl

COPY README.md LICENSE ./
COPY scripts/ scripts/
COPY dwarl/src/ dwarl/src/
COPY ros/src/ ros/src/
ENV CMAKE_BUILD_PARALLEL_LEVEL=2
RUN --mount=type=cache,target=/root/.cache/rattler \
    pixi run --locked build && \
    pixi shell-hook --as-is --environment default --shell bash --no-completions --change-ps1 false \
      > /opt/dwarl-activate.sh

ENV DWARL_PREBUILT_ROS=1 HOME=/tmp/home ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/dwarl-entrypoint
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/dwarl-entrypoint"]
CMD ["--help"]
