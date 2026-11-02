# Compare controllers

**Goal:** compare DWARL, MPPI and DWB on identical Gazebo cases.
Use a checked Gazebo integration and an explicitly selected checkpoint.

```bash
pixi run test --run runs/<run> --checkpoint <iteration> \
  --suite validation --sim gazebo --controllers dwarl,mppi,dwb
```

The default cap is 32 cases per controller. For a shorter diagnostic workload,
add `--set test.max_cases=3`. Comparisons run sequentially.

```bash
pixi run dwarl report --open
```

Compare success, collision and clearance as well as time/distance. Inspect
individual outcomes and keep workload, robot limits and stack settings matched.
Use validation for tuning, then freeze parameters before held-out testing.

Add `--profile-compute` for measured controller/inference latency, CPU, RSS and
simulation real-time factor. Run compute benchmarks on an otherwise idle machine.

See [controller configuration and measurement details](results.md#gazebo-controller-and-compute-comparisons).
