# Test results

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
