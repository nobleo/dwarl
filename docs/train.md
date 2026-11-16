# Train a policy

**Before starting:** select and check a robot with [setup](integrate-robot.md).
Configure its Gazebo adapter first if you intend to use Gazebo with this run.

## Start with validation enabled

```bash
pixi run train --preset navigation
```

The starter recipe uses 64 environments, a 3,000-iteration budget and validation
every 200 iterations. These are starting settings, not a navigation-quality guarantee.
Reduce the environment count for a smaller machine:

```bash
pixi run train --preset navigation --num-envs 8
```

The CLI shows model preparation, dataset creation/reuse, training progress and
the completed checkpoint. Data and robot assets are frozen into the run.

## Inspect or resume

```bash
pixi run dwarl runs list
pixi run dwarl runs watch
pixi run train --resume runs/<run> --max-iterations 5000
```

The resume budget is **5,000 total iterations**. Resume uses the saved recipe.
Use a new run to change the robot, observations or rewards.

Add `--detached` for background training or `--show` for the simulator viewer.
`--show` trains in one unsharded world, which is much slower. To look at a headless run
instead, run `pixi run dwarl runs view` from a second terminal: a viewer shows the first
simulation shard while training continues. Repeat the command or close the window to
return to full speed.
[Monitoring and resume](monitor.md) covers logs, TensorBoard and interruption.

??? info "Presets and configuration"

    `smoke` is a two-iteration pipeline check; `navigation` explicitly enables
    periodic validation. Plain `pixi run train` retains the normal defaults with
    intermediate checks disabled. Presets are recorded in the run.

    Inspect the resolved settings with `--print-config` or inputs with `--dry-run`.
    [Configuration](configuration.md) explains YAML, overrides and early stopping.

**Next:** [evaluate a checkpoint](test.md). The CLI prints a pinned test command.
