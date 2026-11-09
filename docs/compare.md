# Compare controllers

**Goal:** compare DWARL, MPPI, DWB and Regulated Pure Pursuit (RPP) on identical Gazebo cases.
Use a checked Gazebo integration and an explicitly selected checkpoint.

```bash
pixi run test --run runs/<run> --checkpoint <iteration> \
  --suite validation --sim gazebo --controllers dwarl,mppi,dwb,rpp
```

The default cap is 32 cases per controller. For a shorter diagnostic workload,
add `--set test.max_cases=3`. Comparisons run sequentially.

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
