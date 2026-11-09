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

## Use

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
