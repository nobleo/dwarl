# Decision

**Do not claim an improvement or keep training until 95%.** The observed 40.6% on workload A and 81.3% on workload B are not a matched comparison. Their 40.7-percentage-point difference cannot be attributed to the reward change. Episode counts, collisions and uncertainty are also missing.

Stop this training iteration: two comparable trials without useful gain exhaust a two-trial no-improvement patience. If patience was not previously declared, explicitly adopt this stopping decision now rather than claim it was predeclared. Retain the previous verified policy and preserve both trial artifacts. The new reward trial is unproven, not demonstrated worse. A 95% aspiration is not an open-ended stopping rule.

## Bounded next check: evaluation only

Allow **zero additional training trials** and at most **four sequential Gazebo validation runs**: two matched parent/candidate pairs, 32 balanced cases each (128 episode attempts maximum). Use the same frozen case IDs, geometry, starts, order and seeds for every run. If fewer cases are available, record the actual denominator and revise the acceptance threshold before running anything. Do not change the case cap between repeats.

Before evaluation, inspect both runs' manifests/contracts and existing `report/README.md`, `summary.json`, `episodes.json`, failure `trace.json`/`compute.json`, parameter and behavior-tree snapshots, and `launch.log`. Freeze checkpoint hashes, source revision, dataset, robot footprint/sensors/command and motion limits, shared safety/recovery configuration, scoring and runtime limits. Prefer saved best checkpoints; ensure `best` cannot change during evaluation. The commands below assume both runs reference the same validation dataset and selector; matching the cap alone does not establish matching workloads. Resolve any mismatch first.

Hypothesis: the changed reward improves navigation success on identical validation cases without sacrificing safety. Predeclare a useful gain of at least **one additional success out of 32 (3.125 percentage points) in each matched repeat**, no increased collision rate, and no worsening of the existing clearance guardrails. Report exact successes/completed episodes, missing or interrupted episodes, collisions, clearance, termination reasons and per-case changes. This small threshold is a screening criterion, not statistical proof. If evidence remains ambiguous after the fixed budget, retain the parent and stop; do not extend training or loosen scoring.

## Commands for the proposed check — not executed

Replace placeholders with verified paths. Source the robot's workspace as required, in addition to these ROS/DWARL setups. Use one frozen tuning YAML for both policies; keep compute profiling disabled and evaluate DWARL only.

```bash
source /opt/ros/jazzy/setup.bash
source /path/to/robot_ws/install/setup.bash
source /home/bramo/repos/dwarl/ros/install/setup.bash

# First matched pair; run sequentially and record each emitted test directory.
dwarl test --run <parent-run> --checkpoint best --suite validation --sim gazebo \
  --controllers dwarl --set test.max_cases=32 \
  --set test.controller_params=/path/to/frozen-tuning.yaml
uv run python scripts/audit_comparison.py <parent-validation-1-directory>

dwarl test --run <reward-trial-run> --checkpoint best --suite validation --sim gazebo \
  --controllers dwarl --set test.max_cases=32 \
  --set test.controller_params=/path/to/frozen-tuning.yaml
uv run python scripts/audit_comparison.py <candidate-validation-1-directory>

# Second matched pair: identical cases and configuration, new output directories.
dwarl test --run <parent-run> --checkpoint best --suite validation --sim gazebo \
  --controllers dwarl --set test.max_cases=32 \
  --set test.controller_params=/path/to/frozen-tuning.yaml
uv run python scripts/audit_comparison.py <parent-validation-2-directory>

dwarl test --run <reward-trial-run> --checkpoint best --suite validation --sim gazebo \
  --controllers dwarl --set test.max_cases=32 \
  --set test.controller_params=/path/to/frozen-tuning.yaml
uv run python scripts/audit_comparison.py <candidate-validation-2-directory>
```

Also compare saved manifests and case lists **across all four directories**; a per-directory audit alone does not establish cross-run comparability. Infrastructure failures or protocol mismatches invalidate the comparison and do not justify more training. Stop and report the blocker within the budget.

Only select the candidate if both comparable Gazebo repeats satisfy the predeclared criteria. Freeze the selected configuration before one final held-out test; keep held-out outcomes out of tuning. Record the rejected/unproven alternatives and stop reason: **two non-improving comparable trials; training patience exhausted**. No experiments or source edits were performed for this evaluation.
