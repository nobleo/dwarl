# Compare controllers

**Goal:** compare DWARL, MPPI, DWB and Regulated Pure Pursuit (RPP) on identical Gazebo cases.
Use a checked Gazebo integration and an explicitly selected checkpoint.

```bash
pixi run test --run runs/<run> --checkpoint <iteration> \
  --suite validation --sim gazebo --controllers dwarl,mppi,dwb,rpp
```

The default cap is 32 cases per controller. For a shorter diagnostic workload,
add `--set test.max_cases=3`. Comparisons run sequentially.

For the built-in benchmark layouts, including office and warehouse:

```bash
pixi run dwarl test --suite benchmark --sim gazebo --controllers dwarl,mppi,dwb,rpp
```

Use `--case office` or `--case warehouse` to focus on one layout.

## Clean versus obstructed routes

```bash
pixi run dwarl test --sim gazebo --obstacle-comparison \
  --controllers dwarl,mppi,dwb,rpp
```

This is one evaluation with two sequential passes. The clean pass records Nav2's
first global plan for each case. The obstructed pass places up to three stationary
objects along those recorded routes, absent from the planning map. Layouts are
generated once and stored in `obstacles.json`; every controller gets the same objects.
The same checkpoint, maps, starts, goals, seeds, planner and tuning apply to both passes.

Use `--set test.unmapped_obstacles=1` to change the target count. Placement can reject
obstacles in tight spaces; the report includes actual counts and cases with none.
An incomplete clean pass or missing route prevents the obstructed pass from starting.
Completed navigation failures still count as failures and do not prevent comparison
when route evidence exists.

One report contains both passes, matched success/collision differences, and completion-time
differences for cases successful in **both** passes. The case cap applies per controller
per pass. The default command remains a single pass; this larger workload is opt-in.

Add `--show` to watch Gazebo/RViz while the same scored workflow runs. The web workspace's
“Compare clean and obstructed routes” checkbox launches this same command.

```bash
pixi run dwarl report --open
```

For a specific comparison, pass its directory:

```bash
pixi run dwarl report runs/<run>/tests/<test-id> --open
```

The report starts with an aggregate **Controller comparison**, followed by a
per-case table and individual trajectory/motion plots grouped by case and controller.
`gazebo` labels DWARL; `gazebo-mppi`, `gazebo-dwb` and `gazebo-rpp` label the baselines.
Rates in the HTML table are fractions (`0.75` means 75%).

First check that every controller completed the expected number of cases. Controllers
execute sequentially, so an interrupted report can omit later controllers entirely.
Time/distance averages cover successful episodes only: compare cases both controllers
completed successfully before concluding that one is faster or more efficient.

Compare success, collision and clearance as well as time/distance. Inspect
individual outcomes and keep workload, robot limits and stack settings matched.
Use validation for tuning, then freeze parameters before held-out testing.

Add `--profile-compute` for measured controller/inference latency, CPU, RSS and
simulation real-time factor. Run compute benchmarks on an otherwise idle machine.

Reports are static, per evaluation: there is no cross-run dashboard or automatic
overall controller ranking. See [how to read and copy reports](results.md#open-and-read-the-report).

See [controller configuration and measurement details](results.md#gazebo-controller-and-compute-comparisons).
