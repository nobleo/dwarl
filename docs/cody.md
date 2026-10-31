# Cody walkthrough

This walkthrough exercises inspection, data generation, training, validation,
resume, ONNX preparation, both simulators, reports and deployment.
Examples assume sibling `~/repos/dwarl` and `~/repos/cody-dwarl2` checkouts;
adjust the paths for your installation. Cody must include the `dwarl_robot.yaml`
profile and the updated `cody_dwarl2` launch integration.

## 1. Set up the environment

Install DWARL as described on [Start here](index.md), and build the native stack
using [ROS setup](ros.md). On the existing Cody development machine these builds
are already present. In each new terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/repos/cody-dwarl2/install/setup.bash
source ~/repos/dwarl/ros/install/setup.bash
cd ~/repos/dwarl
source .venv/bin/activate
```

Keep this working directory: `runs/` and `.dwarl/` are local to it.
After modifying Cody's integration, rebuild it once with:

```bash
cd ~/repos/cody-dwarl2
COLCON_DEFAULTS_FILE=/dev/null colcon build \
  --base-paths src/pkgs/cody_dwarl2 --symlink-install \
  --packages-up-to cody_dwarl2 --event-handlers console_cohesion+
cd ~/repos/dwarl
```

Then source the environments again in a fresh terminal.

## 2. Inspect Cody

```bash
dwarl doctor --sim mujoco
dwarl doctor --sim gazebo
dwarl robot check ../cody-dwarl2
dwarl robot view ../cody-dwarl2
```

The viewer shows the simplified **training geometry on a floor**, without a
navigation policy or obstacle course. Inspect wheels, chassis and lidar mounting;
close the window to continue. Use `dwarl test --show` to watch navigation.
The Gazebo doctor checks prerequisites, while an actual test checks full startup.

## 3. Smoke-test training and resume

```bash
dwarl train ../cody-dwarl2 --name cody-smoke \
  --num-envs 2 --max-iterations 2 \
  --set train.save_interval=1 \
  --set train.validation_interval=1 --set train.validation_cases=2 \
  --set train.data.train_maps=1 --set train.data.validation_maps=1 \
  --set train.data.test_maps=1 --set train.data.pairs_per_map=3
dwarl runs list
```

Copy the exact run directory printed by training:

```bash
SMOKE="runs/<your-cody-smoke-run-directory>"
dwarl train --resume "$SMOKE" --max-iterations 4
```

The budget is **four total iterations**, not four extra. Resume uses the latest
completed checkpoint and saved recipe; simulator episodes restart.
This smoke policy checks infrastructure, not navigation competence.

### Why can startup take time?

The first run freezes robot assets and generates maps, valid spawns and A* paths.
Normal defaults request 128 training, 16 validation and 16 test maps with 32 pairs
per map. Generation runs on the CPU before PPO begins. Progress includes map
counts, elapsed time and periodic messages while paths are being planned.
Matching later runs reuse `.dwarl/datasets/`; changes to geometry, seed or data
settings can require a new cache.

Increase generation parallelism with `--set train.data.workers=4` if CPU and
memory allow. The default is one worker. The small command above is the quickest
first-run check; it deliberately uses a different dataset from normal training.

## 4. Exercise simulator testing

```bash
dwarl test --run "$SMOKE" --case doorway --sim mujoco --show --set test.max_cases=1
dwarl test --run "$SMOKE" --case doorway --sim gazebo --show --set test.max_cases=1
dwarl test --run "$SMOKE" --case doorway --sim both --set test.max_cases=1
```

ONNX export is automatic. DWARL starts and stops the managed Gazebo/Nav2 stack;
do not launch the office first. Expect navigation failures from the smoke policy.
To check interruption handling, press Ctrl+C during a test, inspect its partial
report, and run the command again. Interrupted episodes are excluded from completed
navigation metrics.

## 5. Train a navigation policy

```bash
dwarl train ../cody-dwarl2 --name cody-navigation --num-envs 64 --max-iterations 3000
```

Reduce `--num-envs` for a smaller machine; use `--device cpu` or `--device cuda`
to override automatic device selection. Add `--show` to watch training.
3,000 iterations is a starting budget, not a guarantee of good navigation.
Set `RUN` to this new run's printed directory:

```bash
RUN="runs/<your-cody-navigation-run-directory>"
```

### Monitor training

TensorBoard is included in the Python environment. In a second activated terminal:

```bash
cd ~/repos/dwarl
tensorboard --logdir runs --host localhost --port 6006
```

Open <http://localhost:6006>. To focus on one run, pass
`--logdir runs/<run>/checkpoints`. Event files start appearing when training begins,
not during data generation. Compare reward, episode length, losses and throughput
under the available Scalars tags. Reward alone does not establish navigation
quality: use validation and scored test reports too.

For background training add `--detached` when starting a run. Inspect its state
with `dwarl runs list`, and stop with `dwarl stop runs/<run>`.
Allow at least one checkpoint to complete before testing resume:

```bash
dwarl train --resume "$RUN" --max-iterations 5000 --detached
```

## 6. Evaluate the full workload

`best` requires completed periodic validation. `latest` selects the latest
completed checkpoint. An iteration number pins an exact checkpoint; use this for
comparisons if training continues and `best` might change.

```bash
dwarl test --run "$RUN" --checkpoint best --dry-run
dwarl test --run "$RUN" --checkpoint best --suite benchmark --sim mujoco
dwarl test --run "$RUN" --checkpoint best --suite validation --sim mujoco
dwarl test --run "$RUN" --checkpoint best --suite test --sim mujoco
dwarl test --run "$RUN" --checkpoint best --suite benchmark --sim both
dwarl test --run "$RUN" --checkpoint best --suite test --sim both
```

Omit `--case` for the full suite. Paired testing uses identical case manifests.
The benchmark cases are `open_space`, `corridor`, `narrow_corridor`, `doorway`,
`l_turn`, `s_turn`, `u_trap`, `dead_end`, `tight_gap` and `clutter`.
Gazebo starts a fresh stack per case and is slower. To repeat one case:

```bash
dwarl test --run "$RUN" --checkpoint best --case doorway --sim both --repeats 3
```

Reports live at `runs/<run>/tests/<test-id>/report/README.md`; also inspect
`summary.json`. See [Results](results.md) for scoring and exit codes.
Choose checkpoints using validation, not the held-out test split.
Gazebo uses spawn-transform odometry, not AMCL localization validation.

## 7. Deploy and drive in the office

```bash
dwarl deploy ../cody-dwarl2 --run "$RUN" --checkpoint best --dry-run
dwarl deploy ../cody-dwarl2 --run "$RUN" --checkpoint best
ros2 launch cody_dwarl2 office.launch.py
```

Deployment adds an immutable bundle and updates the integration's
`selected_policy.yaml`. Historical bundles remain; new weights need no ROS rebuild.
In RViz, use Nav2 Goal to try an open route, a turn, a doorway and a longer office
route. This is an interactive demo; managed tests produce the reproducible scores.

## 8. Check the implementation

```bash
uv run ruff check .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
colcon test --base-paths ros/src --build-base ros/build --install-base ros/install
colcon test-result --test-result-base ros/build
```

Disabling pytest plugin autoload avoids unrelated ROS plugin dependencies.
The native standalone golden replay test can skip without an exported bundle.
