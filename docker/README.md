# DWARL in Docker

Everything needed to train, test and deploy DWARL runs in one image. That includes
ROS 2 Jazzy, Gazebo Harmonic, Nav2 (with the DWB, MPPI and Regulated Pure Pursuit
controllers) and DWARL's Python environment. Nothing is installed on the host apart
from Docker. Runs, caches and logs go to `work/` in this repository.

## Host requirements

- Docker Engine 24 or newer, with the Compose plugin (`docker compose`).
- An X11 or XWayland desktop, for Gazebo, RViz and the MuJoCo viewer.
- Optional but recommended: an NVIDIA GPU with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  PyTorch then trains on CUDA, and Gazebo, RViz and the viewer render with the GPU
  instead of Mesa's software renderer. MuJoCo physics always runs on the CPU, so
  training throughput depends mostly on CPU cores.

Install the NVIDIA Container Toolkit once on Ubuntu (this needs `sudo`):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all ubuntu nvidia-smi   # should list your GPU
```

## Set up and build

```bash
cd docker
./configure-host.sh        # writes docker/.env: your user id, X11 access, GPU if available
docker compose build       # about 15 minutes and 15 GB the first time
```

Re-run `./configure-host.sh` after installing the GPU toolkit, and after logging in
again if windows stop opening (the X authority file changes per login).

## End-to-end demo

```bash
cd docker
docker compose run --rm demo
```

1. **Pick a robot.** The built-in choice is a TurtleBot3 Waffle. To add your own, see
   [Switching robots](#switching-robots).
2. **Pick a training budget**, or reuse a finished run for that robot (offered once one exists).
3. Everything after that runs on its own in the terminal:

| Stage | What happens | Output |
| --- | --- | --- |
| setup | Copies the robot integration to `work/robots/<robot>` and builds two demo worlds (office, warehouse) sized for its footprint | `work/demo_worlds/<robot>/previews/*.png` |
| train | `dwarl train` in MuJoCo, headless, with the GPU for PyTorch when available. One status line shows iteration, success, collision and ETA | `work/runs/<run>`, `work/logs/train-*.log` |
| evaluate | MuJoCo benchmark on DWARL's canonical cases (doorway, corridor, U-trap, clutter, …) | success/collision table, `report/README.md` |
| compare | Gazebo + Nav2, headless: DWARL vs DWB vs Regulated Pure Pursuit on the canonical cases and on the demo worlds, same global planner, map and limits | comparison tables and reports |
| deploy | `dwarl deploy` installs the exported ONNX bundle in the robot integration | `work/robots/<robot>/policies/` |
| showcase | Gazebo and RViz windows open. Nav2 plans a global path from start to goal, and the DWARL controller plugin follows it around obstacles that are **not** on the map | goal reached / failed, per world |

Keys while training runs:

| Key | Action |
| --- | --- |
| `v` | Show or hide a live MuJoCo view of the training simulations (one shard). It slows training while open, so it is off by default |
| `l` | Print the last lines of the training log |
| `q` | Stop training early. Completed checkpoints are kept and the demo continues with them |

During the showcase you can replay each world with `dwb` or `rpp` to see the difference.
Press Enter to close a world and move on.

Budgets (`demo/dwarl_demo/robots.yaml`, presets):

| Preset | Iterations | Purpose |
| --- | --- | --- |
| quick | 400 | Pipeline check. The policy drives but is not presentation quality |
| standard | 1500 | Usable policy for comparisons |
| full | 3000 | DWARL's default budget |

Run without prompts, or only some stages:

```bash
DEMO_ROBOT=turtlebot3 DEMO_PRESET=standard DEMO_YES=1 docker compose run --rm demo
DEMO_ROBOT=turtlebot3 DEMO_PRESET=reuse DEMO_STAGES=deploy,showcase docker compose run --rm demo
```

The `compare` stage starts a fresh Gazebo + Nav2 stack for every case and controller, so
expect about a minute per case per controller. Reports are Markdown files with images, next
to `summary.json` in `work/runs/<run>/tests/<id>/`.

## Switching robots

A robot is selectable when the image contains its integration (a DWARL profile plus a Gazebo
adapter, see [Your own robot](#your-own-robot)) and the demo's robot list names it. The
built-in list is `demo/dwarl_demo/robots.yaml`. Add your robots to `work/robots.yaml`,
which is merged over it, so they stay out of the repository:

```yaml
robots:
  my_robot:
    label: My robot - what it is
    profile: my_robot_dwarl          # ROS package with dwarl_robot.yaml, a path, or a builtin
    baselines: package://my_robot_dwarl/config/baselines.yaml   # optional DWB/RPP tuning
    controllers: [dwarl, dwb, rpp]
    num_envs: 64
    default: true                    # preselect it in the menu
```

## Other commands

```bash
docker compose run --rm shell                          # shell with ROS, DWARL and the venv ready
docker compose run --rm shell dwarl doctor --sim gazebo
docker compose --profile monitor up tensorboard        # http://localhost:6006
```

Inside the container, the working directory is `/work` (`work/` on the host). DWARL
keeps `runs/` and `.dwarl/` there. Run `dwarl` directly instead of `pixi run`; the ROS
packages are prebuilt (`DWARL_PREBUILT_ROS=1`), so nothing is rebuilt. For example:

```bash
dwarl train diffbot --num-envs 32 --max-iterations 200
dwarl runs view                   # toggle a live view of the running training
dwarl test --sim gazebo --controllers dwarl,dwb,rpp --case doorway
```

Python dependencies are resolved from `pyproject.toml` when the image is built (the
native setup pins them in `pixi.lock`). The DWARL source in the image is installed in
editable mode at `/opt/dwarl`. To try Python changes without rebuilding, add a bind mount
of this repository at `/opt/dwarl`. Changes to the C++ controller or the ROS packages
need `docker compose build`.

## Your own robot

Put your robot's ROS packages (a DWARL integration with `dwarl_robot.yaml` and its
description) in the `src/` folder of a colcon workspace, then:

```bash
ROBOTS_WS=$HOME/robots_ws ./configure-host.sh   # stored in docker/.env
docker compose build                            # builds $ROBOTS_WS/src into /opt/robots_ws
docker compose run --rm shell dwarl robot check my_robot_dwarl
```

See [Use your robot in the Docker image](../docs/manual-integration.md#use-your-robot-in-the-docker-image).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `could not select device driver "nvidia"` | Install the toolkit (above) or re-run `./configure-host.sh` to drop the GPU overlay |
| `Authorization required` / windows do not open | Re-run `./configure-host.sh` (refreshes `xhost` and the X authority path) |
| Gazebo or RViz is very slow | No GPU in the container: software rendering. Enable the GPU overlay |
| Files in `work/` owned by root | `HOST_UID`/`HOST_GID` missing: run `./configure-host.sh` |
