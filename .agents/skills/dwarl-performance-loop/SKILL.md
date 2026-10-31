---
name: dwarl-performance-loop
description: Run evidence-led DWARL performance iterations from agent-readable navigation reports. Use for diagnosing failed episodes, tuning DWARL/MPPI/DWB or collision-monitor/recovery settings, choosing checkpoints, planning bounded training trials, and reporting matched navigation or compute comparisons. Includes validation-only selection, repeat checks and iteration early stopping.
---

# DWARL performance loop

## Establish the experiment

Read `AGENTS.md`, `docs/results.md`, the selected run's manifest and contract, and the
existing reports. Prefer a saved best checkpoint to an unexamined latest checkpoint.
Identify the robot's footprint, lidar frame/coverage, velocity and acceleration
limits, command interface, control rate and simulator adapter.

Create a decision record containing:

- checkpoint hash and source revision;
- validation case/workload identity and intended held-out workload;
- measured failure, hypothesis and parameter/code change;
- primary success criterion plus collision/clearance guardrails;
- repeat count, maximum trials, minimum useful improvement and no-improvement patience.

Small screening runs are useful, but include different map types/maps and repeat a
promising candidate before selecting it. A one-case smoke run establishes integration,
not a controller ranking. Keep final test outcomes out of parameter selection.

## Inspect the evidence

Start with `report/README.md`, `summary.json` and `episodes.json`. For each important
failure, read its separate trajectory/motion plots, `trace.json`, `compute.json`,
`nav2_params.yaml`, behavior-tree snapshot and `launch.log`.

Useful distinctions:

- timeout, collision, infrastructure failure and Nav2 success outside scored tolerance;
- requested command versus measured motion, including continued braking after STOP;
- canonical reference path versus the actual Nav2 global plan;
- Nav2 TF pose versus ground-truth pose, with their timestamps;
- stopping/latching versus progress timeout, failed backup and repeated replanning;
- lidar-visible obstacles versus blind sectors; a configured rear stop zone does not
  create rear sensor coverage;
- limiter/intervention counters versus a configuration that never actually activated.

Cite exact paths and simulation/wall timestamps. Present hypotheses as hypotheses.
Use `scripts/audit_comparison.py` to check workload/order, completed counts, shared
Nav2 parameters, artifact presence and consistent terminal measurements.

## Make a targeted change

Use a tuning YAML through `--set test.controller_params=/path/to/tuning.yaml`.
The workflow freezes the file and its hash at test start. Controller sections are
`dwarl`, `mppi`, `dwb`; shared sections include `safety`, `progress_checker`,
`goal_checker`, `motion_limits` and `recovery`. See `docs/results.md` for semantics.

Examples of hypothesis-driven interventions:

- DWB retreating in inflated free space: examine obstacle versus resolution-scaled
  path/goal costs before changing sample counts or disabling obstacle rejection.
- MPPI cutting corners: inspect horizon relative to local map size, path alignment,
  near-collision costs and trajectory sampling. More samples alone may not help.
- DWARL trapped by monitor STOP: compare latch/recovery with monitored escape commands;
  preserve the independent collision monitor and score actual collisions.
- Collision after a zero command: inspect measured deceleration and swept footprint;
  evaluate predictive slowing rather than merely shrinking safety margins.
- Borderline goal mismatch: inspect TF/ground-truth error. A stricter internal goal
  checker can add margin while the scored success threshold remains unchanged.

Keep the shared stack and runtime limits matched for controller comparisons. If a
policy-specific intervention differs, name it. Separate tuning effects from changes
to measurement/scoring; rerun a baseline when the protocol changes.

```bash
dwarl test --run <run> --checkpoint best --suite validation --sim gazebo \
  --controllers dwarl,mppi,dwb --profile-compute \
  --set test.max_cases=8 --set test.controller_params=/path/to/tuning.yaml
uv run python scripts/audit_comparison.py <test-directory>
```

MPPI/DWB are Gazebo-only. Source the robot's ROS workspace and DWARL's workspace.
Use the same frozen cases for repeats, not a changed cap/repeat setting that silently
selects a different workload. Run experiments sequentially and log every candidate.

## Bounded training changes

Do not mutate a historical run's recipe to make resume accept a changed experiment.
Use a new run and the same dataset. Re-evaluate the parent on the current validation
selection; historical rates from a different selector/workload are not a baseline.

For the recurring clearance-penalty experiment:

```bash
uv run python scripts/iterate_policy.py <parent-run> \
  --weights 7.5 10 --max-trials 2 --iteration-patience 2 \
  --max-iterations 250 --validation-interval 50 --training-patience 2 \
  --validation-cases 32 --min-delta 0.03125
```

This is a targeted, fresh-training sweep, not a universal optimizer or warm start.
It writes configurations, validation evidence and stop reasons; promotes only a
success improvement without increased collision rate; and preserves the parent.
For other hypotheses, use ordinary `dwarl train --set ...` with explicit budgets.
Validate a promising new policy in Gazebo twice before switching the selected model.

Stop after the declared number of non-improving trials. If uncertainty dominates a
small gain, repeat the candidate rather than silently relaxing the acceptance rule.
Retain the previous verified configuration when no meaningful improvement is found.

## Freeze and communicate

Run the selected configuration on held-out test cases once tuning is finished.
Record exact successes/episodes, collisions, termination reasons, per-case behaviour,
hardware, units and raw-data links. Plot individual episodes, not large collages.
Latency percentiles pool received calls; CPU/RSS belong to processes, not just the
plugin. Profiling and input serialization add overhead. Keep simulation, standalone
ONNX replay and physical-robot claims separate.

The final response should state the selected configuration, rejected alternatives,
evidence, limitations and iteration stop reason. Preserve scripts and compact decision
records in the repository; commit verified changes when authorized.
