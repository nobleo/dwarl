# DWARL in Docker

Docker packages the **same locked Pixi environment** as local development: ROS 2
Jazzy, Gazebo Harmonic, Nav2, MuJoCo and DWARL. Use the same `setup`, `train`,
`test`, `deploy` and `report` commands and the same robot profiles and configuration.
There is no separate Docker workflow or robot registry.

Runs, caches and configuration persist in the repository's `work/` directory,
mounted at `/work`. Set `DWARL_WORK` to use another host directory.

## Host requirements

- Linux x86-64 with Docker Engine 24+ and the Compose plugin.
- X11 or XWayland for Gazebo, RViz and MuJoCo windows.
- Optional NVIDIA GPU with the NVIDIA Container Toolkit for PyTorch and rendering.
  MuJoCo physics runs on the CPU.

Install the GPU toolkit following
[NVIDIA's instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
then configure and check it:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all ubuntu nvidia-smi
```

## Build and select a robot

```bash
cd docker
./configure-host.sh
docker compose build
docker compose run --rm dwarl setup
```

`configure-host.sh` writes `docker/.env` with your user ID, X11 access and GPU
configuration. Re-run it after installing the GPU toolkit or logging in again
if windows stop opening. The image runs `pixi install --locked`; dependencies
are declared in `pyproject.toml` and pinned in `pixi.lock`.

For your own robot, provide its colcon workspace before building:

```bash
ROBOTS_WS=$HOME/robots_ws ./configure-host.sh
docker compose build
docker compose run --rm dwarl robot check my_robot_dwarl
docker compose run --rm dwarl train my_robot_dwarl --preset smoke
```

Robot sources are copied to `/opt/robots_ws/src` and built into
`/opt/dwarl/.pixi/robot/`. The entrypoint activates the robot and DWARL overlays.
The selected profile is remembered through the normal workflow.
Declare missing ROS dependencies through `pixi add --feature ros ros-jazzy-<pkg>`
on the host, commit `pyproject.toml` and `pixi.lock`, then rebuild.

## Train, test and deploy

```bash
docker compose run --rm dwarl train --preset navigation
docker compose run --rm dwarl runs watch
docker compose run --rm dwarl test --suite validation
docker compose run --rm dwarl deploy
```

Use `--max-iterations` and ordinary `dwarl.yaml` settings for training budgets.
Use `--run` and `--checkpoint` to select existing evidence. To toggle a viewer
on a headless training in an existing container:

```bash
docker exec -it <container> dwarl-entrypoint dwarl runs view
```

## Watch navigation and compare controllers

The office and warehouse are built-in benchmark layouts. `--show` opens Gazebo
and RViz, drives the selected routes, and records standard scored results:

```bash
docker compose run --rm dwarl test --sim gazebo --case office --show
docker compose run --rm dwarl test --sim gazebo --case warehouse --show --controllers rpp
```

For a clean pass followed by frozen obstacles on the recorded Nav2 routes:

```bash
docker compose run --rm dwarl test --sim gazebo --obstacle-comparison \
  --controllers dwarl,mppi,dwb,rpp
```

Both passes share the checkpoint, cases and navigation settings; every controller
gets identical obstacle layouts. One report includes both passes and paired changes.
Add `--case office` for a shorter comparison. Set `test.planner=smac2d` or
`test.controller_params=/work/tuning.yaml` through `--set` when needed.

```bash
docker compose run --rm dwarl report
```

Open the generated `work/runs/<run>/tests/<id>/report/index.html` on the host.
Navigation failures and infrastructure failures remain distinct; a visual run
uses the same scoring as a headless test.

## Other commands

```bash
docker compose run --rm shell
docker compose run --rm dwarl doctor --sim gazebo
docker compose --profile monitor up tensorboard
```

TensorBoard is at <http://localhost:6006>. Inside the shell, run `dwarl` directly:
the entrypoint has activated Pixi. ROS packages are prebuilt
(`DWARL_PREBUILT_ROS=1`).

The source is installed editable at `/opt/dwarl`. Bind-mount individual source
directories to try changes; mounting the whole repository there would hide the
locked environment and ROS builds. Dependency and native/ROS changes require a rebuild.

## Migrating from the former demo

The `demo` service, `DEMO_*` variables and `work/robots.yaml` registry have been
removed. Existing `work/runs` and reports remain usable with `--run`. Select the
profile with `setup` or pass it to `train`. Existing `work/demo_worlds` datasets
can still be tested with `--set test.dataset=/work/demo_worlds/<robot>`, but new
benchmarks automatically include office and warehouse.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| GPU driver could not be selected | Install the NVIDIA toolkit or re-run `configure-host.sh` without the GPU overlay |
| Windows do not open | Re-run `configure-host.sh` to refresh X authority; check X11/XWayland |
| Gazebo or RViz is slow | Confirm hardware rendering; software rendering is much slower |
| Robot ROS package missing | Set `ROBOTS_WS`, add its dependencies through Pixi, rebuild |
| Navigation startup or heartbeat failures | Read `launch.log`, check `dwarl doctor --sim gazebo`, and avoid competing simulator workloads |

See [robot integration](../web/docs/manual-integration.md#use-your-robot-in-the-docker-image)
and [result interpretation](../web/docs/results.md).
