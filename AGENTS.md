# Working on DWARL

DWARL is a robot-neutral differential-drive policy toolkit. Keep the main workflow
`dwarl train`, `dwarl test`, `dwarl deploy`. Optional experiments belong behind
explicit configuration; default tests run DWARL only, at most 32 balanced cases,
with compute profiling and intermediate training checks disabled.

## Read the relevant project skill

- `.agents/skills/dwarl-robot-setup/SKILL.md`: bring a robot model, sensor and command
  contract into training and deployment.
- `.agents/skills/dwarl-performance-loop/SKILL.md`: analyze reports, tune controllers
  and safety, run bounded policy trials, and verify improvements.

These files are portable agent instructions; a harness can load them directly.
The deterministic tools they reference are normal repository scripts.

## Environment: always pixi

Run everything through pixi: `pixi run <task>`, `pixi run dwarl ...`, `pixi run python ...`
or inside `pixi shell`. Never `source /opt/ros/...`, `source .../install/setup.bash`,
`uv run` or `.venv/bin/activate`. Activation (`scripts/pixi_activate.sh`) overlays the robot
workspace and DWARL's `ros/install`, and adds `$DWARL_ROBOT_WS/src` to `DWARL_ROBOT_PATH`.

- `default` env: training + ROS 2 Jazzy/Nav2/Gazebo (`dwarl test --sim gazebo` launches the
  stack itself). `train`: slim, no ROS. `docs`: mkdocs.
- Users start with `pixi run setup` (`dwarl setup`): it asks for the robot repository, saves
  `.dwarl/setup.env`, builds and runs the Gazebo doctor. Non-interactive:
  `pixi run dwarl setup --robot-ws <path> --packages <pkg> --yes`.
- Tasks: `setup`, `train`, `test` (= `dwarl test`, builds first), `deploy` (builds first), `pytest`,
  `lint`, `docs`, `build` (DWARL `ros/`), `build-robot` (saved/`$DWARL_ROBOT_WS`, limited to
  `$DWARL_ROBOT_PACKAGES`, into `.pixi/robot/`), `ros-test`, `doctor`.
- Gazebo runs load the robot overlay and `ros/install` at run time (`dwarl.ros_env.overlay`), so a
  build earlier in the same `pixi run` is picked up.
- Add dependencies with `pixi add --feature ros ros-jazzy-<pkg>` or `pixi add --pypi <pkg>`;
  commit `pixi.lock`.
- Before attributing a Gazebo result to a change, check `compute.json` real-time factor and
  `launch.log` for lifecycle "heartbeat" failures, and read the warnings printed by
  `dwarl doctor --sim gazebo` / `dwarl test` (leftover DWARL Gazebo, `powersave` governor). Never kill
  a Gazebo the user started themselves. A slow or shared simulator produces `nav2_aborted`/timeouts
  that look like controller regressions. Compare setups on the same workload at the same time.

## Evidence-led work

1. Inspect the current worktree and preserve user changes, historical policies and
   existing runs. The repository is public and robot-neutral: never commit robot- or
   customer-specific names, paths, maps, parameters or results; examples must work with
   any robot (use the builtin `diffbot` or the `tests/robots/testbot` fixture).
2. Freeze the checkpoint, robot contract, case IDs, stack configuration and source
   revision. Select on validation; reserve held-out test cases for the final report.
3. Read `summary.json`, `episodes.json`, individual `trace.json`/`compute.json`,
   parameter snapshots and launch logs before proposing a change. Separate observed
   facts from hypotheses. A finished process is not successful navigation.
4. Record a hypothesis, intervention, measured acceptance criteria and trial budget.
   Change one interpretable factor at a time or explicitly label a grouped change.
   Include collision/clearance outcomes and compare safety/recovery as well as the
   controller. Do not loosen scoring to turn near misses into successes.
5. Repeat promising configurations on the same validation cases. Record rejected
   variants too. Stop when the predeclared no-improvement patience or budget expires.
   An inactive intervention counter means different outcomes cannot be attributed
   to that intervention.
6. Run Gazebo experiments sequentially. Run training separately from final compute
   benchmarks. Do not rebuild a controller shared library while a live worker uses it.
7. Audit evidence with `pixi run python scripts/audit_comparison.py <test-directory>`.
   Freeze selected parameters before held-out testing. Fixing a measurement protocol
   requires fresh comparable evidence, not silently mixing old and new counts.
8. Store reusable scripts, robot-neutral instructions, parameter candidates and
   decision records in the repository. Raw runs and models remain in ignored artifact
   directories; curate compact evidence for published studies. Commit verified,
   effective changes incrementally when authorized, with the relevant checks recorded.

## Training iterations

Resume restores the saved recipe; use a new run for changed rewards or observations.
`scripts/iterate_policy.py` runs a bounded clearance-reward sweep on the parent's
dataset, remeasures a comparable validation baseline, and preserves the parent.
It starts fresh training, not a warm start. Its example defaults are 50-iteration
checks, patience 2, a 250-iteration budget and at most two trials. These are aggressive
experiment settings, not project defaults or a guarantee of convergence.

Only completed comparable validation checks affect early stopping. A policy must
improve validation success without increasing collision rate before promotion, then
be verified in Gazebo. Keep checkpoint-hash training-check sidecars with copied runs.

## Verification

```bash
pixi run pytest   # pytest -q with OMP/MKL threads = 1 and plugin autoload disabled
pixi run lint     # ruff check .
git diff --check
```

For ROS changes, run `pixi run build` and `pixi run ros-test` (or focused:
`pixi run pytest -q ros/src/pkgs/dwarl_bringup/test/test_nav_params.py`).
Build docs with `pixi run docs` (`mkdocs build --strict`).
Run focused checks after subsequent changes; repeat broad checks when warranted.

Reports need units and measurement boundaries: wall-clock latency, simulation elapsed
time, CPU 100% per logical core, sampled RSS, and received telemetry count. Controller
server CPU includes the local costmap/ROS work; Gazebo is separate. Simulator stopping
behaviour and desktop ONNX timing are not measurements of physical robot performance.
