# ROSCon 2026 navigation study

This directory preserves parameter candidates and compact decision evidence.
Raw simulation runs and policy weights remain in ignored artifact directories.
The robot-specific parameters here are study inputs, not general toolkit defaults.

## Protocol

- Parent run: `20261031-120652_navigation_b2ce75`, selected iteration 200.
- Checkpoint SHA-256: `e458cb05f1066780e7e3c84b39c61bdc254d79b076159b8bd9d763098f48f808`.
- Eight-case validation workload:
  `9877a00f2028c74201379ddd85794aa4e0d1890440f7f278749d29cee5c47a55`.
- Four map types, seven maps; identical case IDs across controllers and repeats.
- Scored goal tolerance stays 0.30 m; internal Nav2 goal tolerance is tightened
  to 0.25 m after inspecting TF/ground-truth differences.
- Terminal scoring, trace and diagnostics use one frozen ground-truth pose.
  Earlier runs preceding that measurement fix are diagnostic evidence only.
- Final plan: 12 held-out cases per controller after policy/configuration selection.
  Held-out evaluation has not started at this record's creation.

## Validation decisions so far

| Experiment (test directory prefix) | DWARL successes / collisions | MPPI | DWB |
| --- | --- | --- | --- |
| Refined reverse-enabled (`152132`) | 5/8 / 0 | 7/8 / 0 | 7/8 / 0 |
| Forward-only (`153559`) | 4/8 / 1 | 7/8 / 0 | 6/8 / 0 |
| Predictive slowing (`155048`) | 7/8 / 0 | — | — |
| Refined repeat (`155521`) | 5/8 / 1 | 7/8 / 0 | 7/8 / 0 |
| Predictive slowing verification (`161005`) | 6/8 / 0 | 7/8 / 0 | 7/8 / 0 |

Full directory IDs are in `records/` and the parent run's `tests/` directory.
All observed collisions remain counted. Zero recorded DWARL reverse-limit
interventions prohibit attributing the forward-only run's outcome to clipping.

Predictive slowing (`approach_time: 1.2 s`) met the screening criterion of two
additional successes without an increased collision count. All-controller
verification completed in `20261031-161005_b793fe5c` and passed the consistency
audit. DWARL's repeat improved by one success over both non-approach runs, not
the two seen in screening: the full screening effect was not reproduced.
MPPI and DWB retained 7/8 with zero observed collisions. The hypothesis follows a measured collision after a
zero command while the simulated robot was still moving; this is not a physical
braking validation. Predictive slowing remains the leading shared candidate;
final policy selection awaits the bounded training sweep. No more shared-stack
parameter candidates are planned in this study.

Both DWARL failures in the verification were maze timeouts, with five safety
stops and seven recovery actions each. Case `8a3f0f97` ended at 50.1 s simulation
time, 4.154 m from the goal, with minimum clearance 0.0300 m. Case `30b6fc9c`
ended at 52.5 s, 2.367 m from the goal, with minimum clearance 0.0969 m; that
case had succeeded in the first approach run. Neither reached a Nav2 success
outside tolerance. These traces show unresolved stop/recovery behaviour rather
than an infrastructure failure. Unknown/stale command samples remain null.

## Bounded training budget

Use `scripts/iterate_policy.py` to remeasure the parent on 32 matched MuJoCo
validation cases, then try fresh training with proximity weights 7.5 and 10.0.
Maximum two trials, 250 iterations each; check every 50 iterations, training and
trial patience 2. Require at least 1/32 additional successes without increased
collision rate. Verify an accepted policy twice in Gazebo before selecting it.
Stop at patience/budget and retain the parent when no candidate qualifies.

## Measurement boundaries and verification

Validation runs may overlap documentation/unit-test work; their compute samples
are diagnostic, not final benchmark numbers. Final profiling must run separately
from training and other substantial host work. CPU 100% denotes one logical core;
controller-server CPU includes local costmap and ROS; Gazebo is separate.

Source/tooling commit `57407be`: 123 Python tests passed, Ruff passed, strict
documentation build passed, whitespace check passed. ROS packages were built and
the 30-test focused report/compute/Nav2/iteration suite passed before that commit.

`technical_figures.py` draws editable presentation diagrams from the frozen bundle
and implementation. Bundle dimensions and robot limits are explicitly study-specific.
Presentation results must be populated from completed audited final evidence.
