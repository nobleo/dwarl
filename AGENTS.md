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

## Evidence-led work

1. Inspect the current worktree and preserve user changes, historical policies and
   existing runs. Keep `nobleo/dwarl` private unless visibility changes are explicitly
   authorized. Public instructions and examples must work with another robot.
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
7. Audit evidence with `uv run python scripts/audit_comparison.py <test-directory>`.
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
source /opt/ros/jazzy/setup.bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run ruff check .
git diff --check
```

For ROS changes, build from `ros/` with DWARL's `.venv` deactivated, using
`colcon build --packages-select dwarl_controller dwarl_bringup --symlink-install`,
source `ros/install/setup.bash`, and run
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q ros/src/pkgs/dwarl_bringup/test/`.
Build docs with `mkdocs build --strict` in the documented docs environment.
Run focused checks after subsequent changes; repeat broad checks when warranted.

Reports need units and measurement boundaries: wall-clock latency, simulation elapsed
time, CPU 100% per logical core, sampled RSS, and received telemetry count. Controller
server CPU includes the local costmap/ROS work; Gazebo is separate. Simulator stopping
behaviour and desktop ONNX timing are not measurements of physical robot performance.
