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

### Open and read the report

After testing, open the exact result directory printed by the CLI:

```bash
pixi run dwarl report runs/<run>/tests/<test-id> --open
```

`report/index.html` is the browser entry point; `report/README.md` has the same
content as Markdown. `report --open` without a directory selects the latest recorded
result for the selected run, which may be interrupted or belong to another suite.

Read the page in this order:

1. **Controller comparison:** completed/expected counts, success and collision rates,
   and successful-episode time/distance. HTML rates are fractions, not percentages.
2. **Status and identities:** confirm the checkpoint, workload and completion status.
3. **Per-case table:** compare outcomes, goal gap and clearance on the same cases.
4. **Individual episode sections:** inspect trajectory and motion plots, then follow
   trace/episode JSON links for exact measurements. Profiling adds per-episode compute
   tables and plots.

The `gazebo` backend label means DWARL; `gazebo-mppi`, `gazebo-dwb` and `gazebo-rpp` are the
baselines. Plots are grouped by case and backend. This is a static report, without
interactive filtering, a cross-run dashboard or an automatically selected winner.
Successful-only averages can favour a controller that fails the difficult cases;
compare shared successful cases before drawing efficiency conclusions.

#### Running and interrupted evaluations

`episodes.json` is updated as cases finish. The aggregate `summary.json` and report
are written at completion or interruption, and plot rendering takes additional time.
Controllers run sequentially: an interrupted report may contain a complete first
controller, a partial second controller and no third controller. Missing episodes
have no measured outcome; they must not be counted as successes or collisions.
Check completion counts before comparing rates. Missing compute sections are expected
when `--profile-compute` was not enabled.

#### Copy a remote result for local viewing

Copy the **whole test directory**, rather than just the HTML file: its images and JSON
links point to sibling directories. For example, from your local repository:

```bash
mkdir -p runs/<run>/tests/<test-id>
rsync -a user@host:/path/to/dwarl/runs/<run>/tests/<test-id>/ \
  runs/<run>/tests/<test-id>/
pixi run dwarl report runs/<run>/tests/<test-id> --open
```

A copy made during execution is only a snapshot and may lack the final report.
Repeat the copy after completion to obtain the final results. Opening a remote browser
through `--open` requires a graphical session on that machine; local copying avoids this.

### Gazebo controller and compute comparisons

```bash
dwarl test --checkpoint best --sim gazebo --controllers dwarl,mppi,dwb,rpp --profile-compute
```

This runs the same selected cases sequentially for each controller: the default
cap means **32 cases per controller**, or 96 executions for all three. For a quick
check add `--set test.max_cases=3`. MPPI, DWB and Regulated Pure Pursuit (`rpp`) require Gazebo;
the default remains DWARL alone. The Nav2 MPPI, DWB and RPP packages are part of the pixi environment.

The robot footprint, velocity/acceleration limits, control rate, planner, goal
tolerance, safety monitor and behavior tree are shared. MPPI uses a differential-drive
motion model. Baseline parameters are starting presets, not claims of optimal tuning.
DWARL has a safety-stop latch that triggers recovery; baseline controllers rely on
Nav2's normal progress/failure detection, so recovery timing can differ.

The report compares success/collision rates and successful-episode time/distance.
Each controller gets independent episode plots and JSON. Rendered controller and
costmap parameters are saved in `gazebo/<controller>/<case-id>/nav2_params.yaml`.

For reproducible baseline tuning, pass `--set test.controller_params=/path/to/tuning.yaml`.
The YAML maps `mppi`, `dwb`, `rpp` and/or `dwarl` to their `FollowPath` parameters, for example
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
costmap:              # local and global costmap
  inflation_radius: 0.9
  cost_scaling_factor: 3.0
  footprint_padding: 0.01
planner:              # parameters of the selected global planner (GridBased)
  cost_travel_multiplier: 3.0
```

`costmap.inflation_radius` replaces the default, footprint radius + 0.25 m. A lower
`cost_scaling_factor` spreads the inflation cost further, so global plans keep away from
walls. That matters for robots whose base frame is far from the footprint's centre (for
example a long tail behind the drive axle), which sweep a large circle when turning in place.
`planner` overrides parameters of the global planner selected with `test.planner`, but not
its plugin.

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

#### Global planner

`--set test.planner=smac2d` uses Nav2's Smac 2D planner instead of NavFn (default) for every
controller in the test. NavFn caps path extraction at about four map widths of half-cell steps,
so long routes on small maps can fail to plan; Smac 2D has no such cap. The planner is recorded
in each episode's diagnostics.

#### Obstacles that are not on the map

```bash
dwarl test --sim gazebo --controllers dwarl,mppi,dwb --set test.unmapped_obstacles=3
```

This adds up to three boxes or cylinders (0.4–0.7 m across) to the Gazebo world of each case,
but not to the Nav2 map. The global plan therefore runs through them and only the local
planner, from its lidar, can avoid them. They are spread along the case's reference path, at
least 1.5 m from start and goal and 2 m apart, and never touch mapped geometry. With all of
them added, the goal stays reachable for the robot's width plus 0.1 m on each side; on tight
maps fewer obstacles are placed. Placement depends only on the case, so every controller
meets the same obstacles. Collisions are scored against mapped and unmapped geometry, and the
trajectory plots show unmapped obstacles in orange. The obstacles are stored per episode as
`unmapped_obstacles` primitives `[type, x, y, yaw, half_x, half_y]` (type 1 = cylinder).

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
pixi run python scripts/audit_comparison.py runs/<run>/tests/<test-id>
```

For a bounded fresh-training clearance-reward sweep on an existing run's dataset:

```bash
pixi run python scripts/iterate_policy.py runs/<parent-run> \
  --weights 7.5 10 --max-trials 2 --iteration-patience 2 \
  --max-iterations 250 --validation-interval 50 --training-patience 2
```

The runner remeasures a matched validation baseline, preserves historical checkpoints,
and records trial configurations, scores and stop reasons in `runs/studies/`.
These aggressive budgets are experiment examples, not training defaults. Promotion
requires improved validation success without increased collision rate, followed by
repeated Gazebo verification. Stop at the declared patience or budget, freeze the
selected policy/parameters, and only then evaluate held-out cases.
