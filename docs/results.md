# Test results

Tests default to at most **32 episodes per simulator**, balanced across map types
and individual maps using the test seed for reproducible selection.
Use `--set test.max_cases=8` for a shorter check. To explicitly evaluate the full
selected suite, use:

```bash
dwarl test --suite test --set test.max_cases=null
```

Each test lives in `runs/<run>/tests/<test-id>/` and records the request, saved contract,
case geometry, exact start poses, checkpoint hash, runtime bundle reference, episodes
and summary. Failed episodes remain results.

Exit codes: `0` passed, `1` completed navigation workload failed, `2` infrastructure
or input failure, `130` interrupted. Infrastructure failures do not count as navigation
success or collision observations. Missing measurements are null.

Ctrl+C preserves the current case's partial trajectory and available behavior trace
with status `interrupted`; it is excluded from completed navigation metrics.
Start with `report/README.md`: a summary table links to individual episode data
and separate images for each simulator. There are no multi-case collages.

## Per-episode behavior reports

### Gazebo controller and compute comparisons

```bash
dwarl test --checkpoint best --sim gazebo --controllers dwarl,mppi,dwb --profile-compute
```

This runs the same selected cases sequentially for each controller: the default
cap means **32 cases per controller**, or 96 executions for all three. For a quick
check add `--set test.max_cases=3`. MPPI and DWB require Gazebo; the default remains
DWARL alone. Install the Nav2 MPPI and DWB packages and rebuild/source the ROS workspace.

The robot footprint, velocity/acceleration limits, control rate, planner, goal
tolerance, safety monitor and behavior tree are shared. MPPI uses a differential-drive
motion model. Baseline parameters are starting presets, not claims of optimal tuning.
DWARL has a safety-stop latch that triggers recovery; baseline controllers rely on
Nav2's normal progress/failure detection, so recovery timing can differ.

The report compares success/collision rates and successful-episode time/distance.
Each controller gets independent episode plots and JSON. Rendered controller and
costmap parameters are saved in `gazebo/<controller>/<case-id>/nav2_params.yaml`.

For reproducible baseline tuning, pass `--set test.controller_params=/path/to/tuning.yaml`.
The YAML maps `mppi`, `dwb` and/or `dwarl` to their `FollowPath` parameters, for example
`mppi: {batch_size: 2000}`. The input YAML is frozen at test start, with its hash,
and rendered parameters are saved per episode. Select settings on validation cases, then freeze them before testing;
keep robot limits and the shared navigation stack matched between controllers.

Optional shared tuning sections are applied identically to every selected controller:

```yaml
goal_checker:
  xy_goal_tolerance: 0.25  # may tighten, never loosen, the bundle's scored tolerance
progress_checker:
  required_movement_radius: 0.2
  movement_time_allowance: 8.0
motion_limits:
  min_linear_velocity: 0.0  # forward-only controller proposals; preserves trained upper limits
safety:
  rotation_sweep: true
  approach_time: 1.2  # optional predictive slowing horizon [s]; validate braking assumptions
recovery:
  through_monitor: true
  retry_count: 1
  backup_required: true
  replan_after_backup: true
```

These are **experimental overrides**, not new defaults or universally optimal settings.
The angular stopping envelope uses the bundle's angular acceleration limit, latency
and footprint margin to cover a finite braking sweep, including near-zero translational
drift. Its braking assumptions must match the robot; simulation results do not measure
physical stopping performance. Translation zones and lidar visibility still matter.
`motion_limits` narrows the runtime command range for all controllers; it does not retrain
the policy. Explicit Nav2 backup recovery remains available and costmap-checked.

`recovery.through_monitor` also routes recovery commands through the external collision
monitor. `backup_required` propagates a blocked backup to outer recovery instead of
claiming a successful retry. The rendered behavior tree is saved next to the Nav2 YAML.
To compare DWARL's latch against monitor-only control, use
`dwarl: {safety_state_topic: ""}`: this disables only the plugin's stop-event latch;
the external collision monitor still guards commands and the progress checker still
requests recovery. Record this difference when reporting comparisons.

Gazebo records a frozen terminal ground-truth pose for scoring and diagnostics, plus
the latest Nav2 TF pose and its timestamp. Spawn-transform odometry can drift relative
to ground truth; a Nav2 success outside the scored tolerance remains a failure.

`--profile-compute` adds `compute.json` per episode with:

- Wall-clock controller-call latency, including failed calls: p50/p95/p99/max.
- DWARL ONNX inference and observation-building timings, separately.
- Calls exceeding the configured control-period budget (not scheduler deadline misses).
- Sampled CPU and RSS memory for controller_server, Gazebo and the remaining stack.
  CPU 100% means one logical core; summed RSS can double-count shared memory.
- Timestamped raw samples, hardware identification and simulation real-time factor.

Profiling excludes launch/startup. The controller-server CPU/RSS includes its local
costmap and ROS work, not just the plugin. Sampling can miss short-lived processes.
Unavailable measurements stay null. Profiling adds overhead, especially while the
first 128 DWARL observations are recorded; compare controllers on an otherwise idle
machine and use unprofiled runs when measuring navigation without instrumentation.

#### Benchmark the model on the target computer

Profiled DWARL episodes also save `gazebo/dwarl/<case-id>/observations.npz`.
Copy that file and the selected policy bundle to the target computer, then run:

```bash
dwarl model-benchmark /path/to/bundle /path/to/observations.npz --threads 1
```

This needs ONNX Runtime, not ROS/Gazebo. It warms up for 100 calls, then measures
1,000 batch-one CPU inferences over the recorded inputs. Use `--threads`,
`--warmup`, `--iterations` and `--output` to control the experiment. The JSON records
model/input hashes, runtime version, hardware, CPU use and latency percentiles.
It measures Python ONNX inference only; deployment also needs observation building,
costmaps, planning and ROS. Gazebo speed on a desktop is not a target-hardware benchmark.

```text
tests/<test-id>/
├── summary.json
├── episodes.json
├── report/README.md
└── episodes/<case-name>-<case-id>/<simulator>/
    ├── episode.json
    ├── trace.json
    ├── trajectory.png
    └── motion.png
```

`episode.json` identifies the case, simulator and exact checkpoint and records
outcome, termination reason, duration, distance travelled, final pose/distance to
goal, path completion, minimum clearance, and observed safety-stop/recovery counts.
Outcomes distinguish success, collision, timeout, stuck, aborted, other termination,
interruption and infrastructure failure. A Nav2 abort is not automatically labelled
"stuck". Measurements unavailable from a backend remain **null**, not zero.

`trace.json` is readable JSON with `schema`, `units`, `sources`, `samples` and
`events`. Samples contain elapsed simulation time, position/heading, commanded and
measured linear/angular velocity, footprint clearance, distance to goal, and
remaining distance/completion along the canonical reference path. Events mark
collision detection, safety stops/resumes, recovery notifications and termination.
Event poses locate the robot at detection; they are not inferred obstacle contact points.

Use the images for the overall behavior and the JSON to inspect exact values:

- **Trajectory:** obstacles, dashed reference, actual path coloured by measured
  speed magnitude, start/goal, final position/heading and event markers.
- **Motion:** aligned time-series of commanded/measured linear speed, turn rate
  and clearance, with corresponding event markers. Signed speed shows reversing.

If a simulator fails before supplying measurements, the episode JSON and error
remain available; the report does not invent a trace or a plot. Raw data are
written before plotting, so a rendering failure does not lose the measurements.

## Measurement conventions

- MuJoCo records each control tick before automatic episode reset. Its command
  is the **rate-limited policy velocity applied to the drive**, with no safety filter.
- Gazebo records ground-truth pose samples with the latest fresh ROS measurements.
  Its command comes from the configured robot command topic: **after collision
  monitoring, or from a direct Nav2 recovery**. It is not the raw network output.
  Measured velocity comes from odometry. Command/velocity become null if no message
  was received within 0.5 seconds of simulation time; the trace labels this convention.
- Path completion and remaining distance use nearest projection onto the **canonical
  reference**, not Nav2's replanned path. They may jump at path crossings and should
  be interpreted together with final goal distance and the actual trajectory.
- Clearance is nonnegative distance to footprint contact/overlap. Zero does not
  encode penetration depth. Source and backend limitations are recorded in the trace.

These richer files are produced by new evaluations; existing report files are not
retroactively populated with measurements they never recorded.

MuJoCo uses physical contacts and geometry-based footprint clearance. Travelled
distance sums displacement at each control tick. Path efficiency is reference length
divided by travelled distance, capped at one, and zero for unsuccessful episodes.

Paired simulator tests must use the same case manifest. Nav2 may calculate a different
global path, so equal worlds and starts do not imply identical controller inputs.

## Report-driven iteration

Repository agent instructions live in `AGENTS.md` and `.agents/skills/`. The setup
skill covers a new robot's contract; the performance-loop skill covers diagnosis,
matched validation, bounded training and final held-out evaluation.

Audit a comparison before using its numbers:

```bash
uv run python scripts/audit_comparison.py runs/<run>/tests/<test-id>
```

For a bounded fresh-training clearance-reward sweep on an existing run's dataset:

```bash
uv run python scripts/iterate_policy.py runs/<parent-run> \
  --weights 7.5 10 --max-trials 2 --iteration-patience 2 \
  --max-iterations 250 --validation-interval 50 --training-patience 2
```

The runner remeasures a matched validation baseline, preserves historical checkpoints,
and records trial configurations, scores and stop reasons in `runs/studies/`.
These aggressive budgets are experiment examples, not training defaults. Promotion
requires improved validation success without increased collision rate, followed by
repeated Gazebo verification. Stop at the declared patience or budget, freeze the
selected policy/parameters, and only then evaluate held-out cases.
