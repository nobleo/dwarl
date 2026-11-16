# Manual integration reference

For guided setup, start with [Integrate your robot](integrate-robot.md).
This reference explains the profile, simulator and runtime contracts in detail.

This is the end-to-end path for a **differential-drive robot with a 2D lidar**.
You bring the robot description and ROS interfaces; DWARL builds the training
model, generates data, trains a policy, evaluates checkpoints and exports an
ONNX bundle for its Nav2 controller plugin.

```text
Your URDF + dwarl_robot.yaml
           ↓
     train → test → deploy
                       ↓
           policy.onnx + deploy.yaml
                       ↓
Nav2 global path → DWARL controller → velocity commands → your base
```

DWARL replaces the **local controller**, not the complete navigation system.
There is no pretrained navigation-quality policy included.

## 1. Prepare your workspace

Use [pixi](https://pixi.sh): it installs Python 3.12, ROS 2 Jazzy and Gazebo Harmonic, so
no system ROS install is needed (Linux x86-64).

```bash
git clone https://github.com/nobleo/dwarl.git
cd dwarl
pixi install
pixi run dwarl doctor --sim mujoco
```

URDF/xacro imports use pixi's xacro; referenced description/mesh packages must be in your
robot workspace. Build the [native ROS packages](ros.md) before Gazebo testing or Nav2
deployment.

For the rest of this guide, set these to **your actual paths**:

```bash
export DWARL_REPO="$HOME/repos/dwarl"
export DWARL_ROBOT_WS="$HOME/repos/my_robot"
export INTEGRATION="$DWARL_ROBOT_WS/src/my_robot_dwarl"
```

Then open a DWARL shell. It activates ROS, your robot workspace (once set up with
`pixi run setup`, below) and DWARL's ROS packages; nothing needs to be sourced:

```bash
cd "$DWARL_REPO"
pixi shell
```

Keep this working directory for DWARL commands: `runs/` and `.dwarl/` are stored
relative to it. `$DWARL_ROBOT_WS/src` is added to `DWARL_ROBOT_PATH` automatically, so
description packages are found from source for model loading; Gazebo launch packages
are built by `pixi run setup`.

### What must already work on your robot?

| You provide | Used for |
| --- | --- |
| URDF/xacro with wheel joints, inertials and lidar link | MuJoCo training model |
| Actual footprint, velocity/acceleration limits and lidar settings | Training/runtime contract |
| `sensor_msgs/msg/LaserScan` and `nav_msgs/msg/Odometry` | Policy input and navigation |
| TF: `map → odom → base_link → lidar` (use your base/lidar names) | Path and scan transforms |
| Base driver accepting `Twist` or `TwistStamped` | Executing velocity commands |
| Map server + localization, or SLAM supplying a map and localization TF | Real-robot Nav2 navigation |
| A robot-only Gazebo launch adapter | Managed Gazebo tests, if desired |

You can train and test in MuJoCo before implementing the Gazebo adapter.
If you want the **same run** to support Gazebo testing, prepare the adapter in
section 6 and set `ros.sim_launch` before starting that run.

## 2. Add a robot integration

Keep the integration in **your robot repository**, not inside DWARL. A small ROS
package is convenient when you also need launch files:

```bash
mkdir -p "$DWARL_ROBOT_WS/src"
cd "$DWARL_ROBOT_WS/src"
ros2 pkg create my_robot_dwarl --build-type ament_cmake --license MIT
mkdir -p "$INTEGRATION/launch"
cd "$DWARL_REPO"
```

Use your project's license when creating its package. You can instead use an
existing robot package. The intended layout is:

```text
my_robot_dwarl/
├── package.xml
├── CMakeLists.txt
├── dwarl_robot.yaml
├── launch/
│   └── simulation.launch.py     # your adapter, needed for Gazebo tests
├── policies/                    # created by dwarl deploy
└── selected_policy.yaml         # created by dwarl deploy
```

In `CMakeLists.txt`, add these before `ament_package()`:

```cmake
install(FILES dwarl_robot.yaml DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
```

Declare runtime dependencies in `package.xml` for your launch files (typically
`launch`, `launch_ros`, `ament_index_python`, `python3-yaml`, `dwarl_bringup` and your robot's
description/simulation packages). Policies will be addressed by their source
path or copied to the target robot; they do not need to be baked into the build.

### Create `dwarl_robot.yaml`

[Download the profile template](examples/dwarl_robot.yaml), copy it into your
integration and replace its example values. It is a schema example, not a model
of your robot.

```bash
cp "$DWARL_REPO/src/dwarl/templates/dwarl_robot.yaml" "$INTEGRATION/dwarl_robot.yaml"
```

The profile describes:

| Section | What to set |
| --- | --- |
| `description` | URDF/xacro URI, base link and xacro arguments |
| `drive` | Left/right wheel joints and MuJoCo actuator parameters |
| `limits` | Linear/angular speed and acceleration limits in SI units |
| `footprint` | Ordered, strictly convex polygon in the base frame, or radius |
| `lidar` | Lidar link and **post-filter** field of view, range and ray count |
| `collision`, `physics` | Training contact geometry and dynamics |
| `ros` | Actual frames, topics, command type and optional simulation adapter |
| `deployment` | Where bundles and the selected-policy file are installed |

Paths may be relative to the profile, absolute, or `package://package/path`.
Use `description.urdf` for a URDF or xacro. MJCF profiles are also supported,
but must satisfy DWARL's internal model naming contract; URDF is the normal
integration route.

<a id="model-conventions"></a>
**Model conventions:** x points forward, y left, z up. Drive wheels must be
aligned laterally, with the left wheel at positive y and both joint axes pointing
along base +y. Wheel radius comes from cylindrical/spherical collision geometry;
set `drive.wheel_radius` if that geometry is absent. Supply sensible inertials.

DWARL simplifies collisions: drive wheels retain their geometry; the chassis
becomes a footprint prism. Casters may need `collision.supports` spheres or
explicit `keep_collision_links`; set these from your model so it rests correctly.
`chassis_z` and support positions are in the base frame, not world coordinates.
`chassis_mass` adds mass, so do not duplicate mass already represented by the URDF.

Lidar mounting is read from the URDF and real scans use live TF. Configure the
scan **after any filtering**, including its useful field of view. The controller
reprojects scans into the trained ray layout; it does not make unseen regions
visible. Match the robot's real sensing and command limits before training.

For the standard bringup, keep `ros.map_frame: map`: some Nav2 global-frame
settings currently remain fixed to `map`. Other global-frame names need a
customized Nav2 configuration.

### Validate the model

```bash
dwarl robot check "$INTEGRATION"
dwarl robot view "$INTEGRATION"
dwarl train "$INTEGRATION" --print-config
dwarl train "$INTEGRATION" --dry-run
```

Check reported mass, wheel radius/separation, lidar pose and spawn height. The
viewer shows the simplified training geometry on a floor. It should remain
supported, rather than fall through the floor or tip because of missing supports.
Close the viewer to continue. This checks the model, not navigation performance.

## 3. Smoke-test the training pipeline

```bash
dwarl train "$INTEGRATION" --name robot-smoke \
  --num-envs 2 --max-iterations 2 \
  --set train.save_interval=1 \
  --set train.validation_interval=1 --set train.validation_cases=2 \
  --set train.data.train_maps=1 --set train.data.validation_maps=1 \
  --set train.data.test_maps=1 --set train.data.pairs_per_map=3
dwarl runs list
```

Copy the printed run directory, then exercise resume and inference:

```bash
SMOKE="runs/<your-smoke-run>"
dwarl train --resume "$SMOKE" --max-iterations 4
dwarl test --run "$SMOKE" --case doorway --sim mujoco --show --set test.max_cases=1
```

The resume budget is **four total iterations**, not four additional iterations.
The smoke policy checks data, checkpointing and export/inference; navigation
failures are expected. Test automatically exports the selected checkpoint to ONNX.

On the first run, DWARL freezes the robot assets and generates maps, spawns and
A* paths before training begins. Progress messages show this work. Normal defaults
use 128/16/16 train/validation/test maps and 32 pairs per map. Matching runs reuse
`.dwarl/datasets/`; changes to robot geometry, seed or data settings can regenerate
it. Set `--set train.data.workers=4` to parallelize generation if resources allow.

## 4. Train and monitor

```bash
dwarl train "$INTEGRATION" --name navigation --num-envs 64 --max-iterations 3000 \
  --set train.validation_interval=200
```

Reduce `--num-envs` for a smaller machine. Device selection is automatic, with
`--device cpu` and `--device cuda` overrides. Add `--show` for a viewer or
`--detached` for background training. The training budget is a starting point,
not a guarantee of navigation quality.

The command above explicitly enables intermediate validation so that the following
`--checkpoint best` examples have selection evidence. Intermediate checks and
early stopping are otherwise disabled by default. Optionally add
`--set train.benchmark_interval=500` for canonical checks or
`--set train.early_stop.enabled=true` for validation-based plateau stopping.
See [check scheduling and patience](configuration.md#optional-intermediate-testing-and-early-stopping).

```bash
RUN="runs/<your-navigation-run>"
```

In a second activated terminal, monitor training:

```bash
tensorboard --logdir runs --host localhost --port 6006
```

Open <http://localhost:6006>. Events appear after data generation when training
begins. For one run, use `--logdir runs/<run>/checkpoints`. Reward, episode length,
losses and throughput help diagnose learning; scored validation establishes how
the policy navigates. See [monitoring and resume](configuration.md#monitor-training).

## 5. Evaluate a trained checkpoint

`latest` is the latest completed checkpoint; `best` requires completed periodic
validation. Use an iteration number to pin a checkpoint if training is still
running and you want stable comparisons.

```bash
dwarl test --run "$RUN" --checkpoint best --dry-run
dwarl test --run "$RUN" --checkpoint best --suite benchmark --sim mujoco
dwarl test --run "$RUN" --checkpoint best --suite validation --sim mujoco
dwarl test --run "$RUN" --checkpoint best --suite test --sim mujoco
```

The canonical benchmark includes `open_space`, `corridor`, `narrow_corridor`,
`doorway`, `l_turn`, `s_turn`, `u_trap`, `dead_end`, `tight_gap` and `clutter`.
Use `--case doorway --show` for one scenario, `--repeats 3` for repeated execution,
or omit `--case` for the full suite. Choose checkpoints using validation, not the
held-out test split.

Read `runs/<run>/tests/<test-id>/report/README.md` and `summary.json`. Exit `1`
means completed navigation failed; exit `2` means an infrastructure/input failure.
Ctrl+C preserves partial results. [Results](results.md) explains the metrics.

## 6. Add Gazebo testing

Skip this section if you only need MuJoCo evaluation and hardware deployment.
DWARL can orchestrate your simulator, but cannot invent a Gazebo model, drive
plugin or sensor bridge from the training profile.

Create `launch/simulation.launch.py` in your integration and set:

```yaml
ros:
  # Keep the other ros settings from your profile.
  sim_launch: {package: my_robot_dwarl, file: simulation.launch.py}
```

The adapter must accept these launch arguments:

| Argument | Required behavior |
| --- | --- |
| `world` | Start Gazebo using the supplied SDF, rather than a fixed robot world |
| `x`, `y`, `yaw` | Spawn the robot at the requested world pose |
| `gui` | `true` shows the GUI; `false` runs headless |

Reuse your existing simulation launch and map these arguments to its own names.
It must start **only the simulator and robot interfaces**, including robot state
publisher, sensors, drive controller and bridges. DWARL starts map server,
localization transform and Nav2. Do not start a second navigation stack.

The adapter contract is:

- Publish `/clock`, the profile's scan and odometry topics, and `odom → base` TF.
- Publish base-to-sensor TF and accept the configured command topic/message type.
- Spawn the Gazebo entity as **`robot`**. Managed scoring currently uses that name
  and `/world/default/dynamic_pose/info`; generated test worlds are named `default`.
- Use odometry starting at zero at spawn. The benchmark applies the requested
  spawn pose as `map → odom`; world-coordinate odometry would apply it twice.
- Preserve the inherited ROS domain and Gazebo partition so managed cases remain isolated.

The included [TurtleBot simulator adapter](https://github.com/nobleo/dwarl/blob/main/ros/src/pkgs/dwarl_bringup/launch/tb3_sim.launch.py)
shows how to include Gazebo and a robot spawn launch. Adapt that pattern to your
model and bridges; it is not a substitute for your robot's simulator. For a base
launch that already accepts the five arguments above, the wrapper can be as small as:

```python
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Replace these names with your existing robot-only simulator launch.
    source = Path(get_package_share_directory("my_robot_simulation")) / "launch/robot.launch.py"
    defaults = {"world": "", "x": "0", "y": "0", "yaw": "0", "gui": "false"}
    return LaunchDescription([
        *(DeclareLaunchArgument(key, default_value=value) for key, value in defaults.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(source)),
            launch_arguments={key: LaunchConfiguration(key) for key in defaults}.items(),
        ),
    ])
```

If your simulator calls them `x_pose`, `world_file`, etc., change the argument keys
passed to that included launch. Its implementation must still satisfy the entity,
odometry, bridge and TF contract above.

Some pitfalls when writing the adapter:

- A ros2_control macro from your description package may clamp wheel velocity commands
  (for example to ±1 rad/s, which is a few cm/s on small wheels). If so, write the
  ros2_control block in the integration instead of including that macro.
- If a caster or omni-wheel is missing from the URDF, add a frictionless collision sphere
  for Gazebo, and a matching `collision.supports` sphere in the profile for MuJoCo.
- `base_footprint` is not necessarily at the robot's centre. Express the footprint,
  supports and lidar pose in the base frame the profile names.
- Gazebo needs the drive and sensor plugins (`gz_ros2_control`, `gpu_lidar`) and bridges
  for `/clock` and the scan topic; odometry and TF come from the diff drive controller.

Add the adapter to the profile **before training** if you plan to use Gazebo:
the run freezes its ROS settings too. If you add or change this interface after
training, start a new run rather than editing the saved contract.

Build the integration after adding its launch file (from `$DWARL_REPO`, in a new terminal):

```bash
pixi run setup              # select my_robot_dwarl; builds it and its dependencies, then checks Gazebo
pixi shell
```

Launch-file and config edits take effect through the symlink install. Gazebo tests and
deployment with a launch adapter prepare the required ROS packages incrementally, both
through pixi task shortcuts and direct CLI commands inside `pixi shell`. You can also
rebuild explicitly with `pixi run build-robot`.

Then test:

```bash
dwarl doctor --sim gazebo
dwarl test --run "$SMOKE" --case doorway --sim gazebo --show --set test.max_cases=1
dwarl test --run "$RUN" --checkpoint best --suite benchmark --sim both
dwarl test --run "$RUN" --checkpoint best --suite test --sim both
```

If the smoke run predates the adapter, create a new smoke run first. `--sim both`
uses the same case manifest for both backends. Gazebo starts a fresh stack per
case and takes longer. Its spawn-transform localization tests the controller,
not AMCL. Ground-truth scoring requires Gazebo Transport 13 / Messages 10 Python
bindings; the generic ROS Pose_V bridge loses model names.

## 7. Export and install the network

There is no separate `dwarl export` command. **Test and deploy automatically export
an exact checkpoint into an immutable ONNX bundle.** Deploy also installs it in
your integration:

```bash
dwarl deploy "$INTEGRATION" --run "$RUN" --checkpoint best --dry-run
dwarl deploy "$INTEGRATION" --run "$RUN" --checkpoint best
```

Copy the installed bundle path printed by deployment:

```bash
export BUNDLE="$INTEGRATION/policies/<printed-bundle-id>"
```

The directory contains `policy.onnx`, `deploy.yaml`, the frozen profile/model and
assets, resolved configuration and provenance hashes. `deploy.yaml` carries the
observation layout, scan preprocessing/history, action mapping, control rate,
limits and ROS settings needed to run that network correctly.

Keep the **whole bundle** together; copying only the ONNX file loses its runtime
contract. Do not edit hashed bundle files. Export validates compatibility, not
navigation quality: deployment does not require all recorded tests to have passed.
Changes to robot geometry/profile may require a new training run; deployment
checks the target against the frozen training model.

`selected_policy.yaml` records the installed bundle's relative path and checkpoint.
The generic Nav2 launch takes `model_dir` explicitly; it does not automatically read
that selection file. Your robot launch can read `selected_policy.yaml` and pass
the resolved path to `navigation.launch.py` to automate selection. Loading new
weights requires a Nav2 restart/reconfigure, not a ROS rebuild.

For that convenience wrapper, [download this launch file](examples/navigation.launch.py)
into your integration's `launch/` directory:

```bash
cp "$DWARL_REPO/src/dwarl/templates/navigation.launch.py" "$INTEGRATION/launch/navigation.launch.py"
```

Set `deployment.launch: {package: my_robot_dwarl, file: navigation.launch.py}`
in the profile and rebuild the integration once. Then, after starting your robot
interfaces and localization, use:

```bash
ros2 launch my_robot_dwarl navigation.launch.py integration:="$INTEGRATION"
```

This wrapper uses the selected-policy file from that directory and includes the
same standard navigation launch described below. For a robot receiving only a
copied bundle, use the explicit `model_dir` launch instead.

### Transfer to the robot computer

Use the [dedicated pixi runtime environment](deploy.md#use-another-robot-computer)
on supported Linux x86-64 targets. It builds and activates the native packages
automatically. The C++ controller uses ONNX Runtime; it does not need the training
Python environment, PyTorch, MuJoCo or a training GPU.

For example, replacing the login and destination:

```bash
ssh robot@robot-host 'mkdir -p ~/dwarl-policies'
scp -r "$BUNDLE" robot@robot-host:~/dwarl-policies/
```

On the robot, set `BUNDLE` to the copied directory's **absolute path**.

## 8. Run with Nav2

### Use DWARL's navigation launch

First launch your robot drivers, state publisher, scan filtering, odometry,
map server and localization (or your SLAM stack). They must provide the interfaces
from section 1. Then, on that same computer:

```bash
ros2 launch dwarl_bringup navigation.launch.py \
  model_dir:="$BUNDLE" use_sim_time:=false
```

Use `use_sim_time:=true` for a simulator. The launch starts controller, planner,
behavior and BT navigator servers, lifecycle management, and a collision monitor.
It reads the bundle to set the robot footprint, scan topic, limits and control
rate. It does **not** start drivers, localization or a map server. Avoid running
it alongside another full Nav2 navigation launch.

The default command path is controller → `cmd_vel_raw` → collision monitor →
your configured command topic. Match `ros.cmd_vel_stamped` to your base driver
or mux. The launch also offers `cmd_vel_topic:=...` and
`cmd_vel_stamped:=true|false` overrides for command routing.

Open RViz with your robot/map display and send a **Nav2 Goal**. Check an open
route, turning and a passage, then inspect motion against the scored simulation
results. The policy controls position/path following; it does not enforce a final
goal heading. The provided goal checker therefore uses a relaxed yaw tolerance.

### Already have your own Nav2 stack?

Keep that launch and load the plugin in its existing controller server instead.
This snippet illustrates the replacement, not a full Nav2 parameter file:

```yaml
controller_server:
  ros__parameters:
    controller_frequency: 10.0  # set to 1 / deploy.yaml control.dt
    controller_plugins: [FollowPath]
    FollowPath:
      plugin: dwarl_controller::DwarlController
      model_dir: /absolute/path/to/complete/bundle
      expected_robot_id: my_robot
      scan_topic: /scan
      scan_timeout: 0.5
      transform_tolerance: 0.2
      inference_threads: 1
```

Match the actual bundle's robot ID, frequency, footprint, frames, scan topic,
goal tolerance and command limits. Retain your global planner, costmaps,
localization, lifecycle setup and command routing. The supplied
[Nav2 configuration](https://github.com/nobleo/dwarl/blob/main/ros/src/pkgs/dwarl_bringup/config/nav2_params.yaml)
and [parameter renderer](https://github.com/nobleo/dwarl/blob/main/ros/src/pkgs/dwarl_bringup/dwarl_bringup/nav.py)
show which values the standard launch derives from the bundle. The controller
already applies the trained action rate limits; account for any additional
velocity smoothing in your stack.

## Use your robot in the Docker image

The [Docker image](https://github.com/nobleo/dwarl/tree/main/docker) builds an optional
robots workspace next to DWARL, so your integration and description packages need no
changes to the DWARL repository. It is the Docker counterpart of `DWARL_ROBOT_WS`:

1. Put your packages in a colcon workspace, for example `~/robots_ws/src/my_robot_dwarl`
   and `~/robots_ws/src/my_robot_description`. Use plain copies, or `git archive` a
   fixed ref of a private repository; symlinks to outside the workspace are not followed.
2. Point the image at it and build:

    ```bash
    cd docker
    ROBOTS_WS=$HOME/robots_ws ./configure-host.sh   # stored in docker/.env
    docker compose build
    docker compose run --rm shell dwarl robot check my_robot_dwarl
    ```

The build copies `$ROBOTS_WS/src` to `/opt/robots_ws/src`, installs dependencies with
rosdep (keys it cannot resolve are skipped), and builds it with colcon. The container
sources `/opt/robots_ws/install/setup.bash`, so `my_robot_dwarl` works as a profile name
and `package://` URIs resolve. Without `ROBOTS_WS`, the empty `docker/robots_ws` is used.
Rebuild the image after changing the packages.

To offer the robot in the Docker demo (`docker compose run --rm demo`), add it to
`work/robots.yaml`, which is merged over the built-in robot list; see `docker/README.md`.

## Troubleshooting the first integration

| Symptom | Check |
| --- | --- |
| Profile/package not found | Pass the exact profile path; run `pixi run setup` (or set `DWARL_ROBOT_PATH`) |
| Multiple integrations found | Pass the intended `dwarl_robot.yaml`, not the workspace root |
| Robot falls or tips in the viewer | URDF inertials, wheel collision shapes and support/caster geometry |
| Startup is slow | Generation progress; matching caches are reused; reduce smoke data or increase workers |
| `best` is unavailable | Complete periodic validation, or use `latest` for an infrastructure test |
| Gazebo readiness timeout | Case `launch.log`, clock, scan, odom, TF, entity name `robot` and installed adapter |
| Many `nav2_aborted`/timeouts, lifecycle "heartbeat" errors | `compute.json` `real_time_factor`; leftover DWARL Gazebo and `powersave` warnings from `dwarl doctor --sim gazebo`; compare only runs made under the same load |
| Robot appears at the wrong map pose | Odometry origin at spawn and duplicate `map → odom` publishers |
| Controller refuses/stops on scan input | Scan age, frame TF, filtering and configured topic |
| Nav2 runs but the base does not move | Output topic, `Twist` vs `TwistStamped`, mux routing and collision-monitor state |
| Bundle incompatible with target | Profile or referenced geometry changed after training |

For all CLI switches and resolved defaults, see the [generated reference](reference.md).
