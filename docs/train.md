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

## Obstacles that are not on the map

`recipe.env.surprise_obstacles` adds random boxes and cylinders to the reference path of a
share of training episodes (`probability`, 0 by default). The map and the path stay as they
were, so the robot only sees them in its lidar, like obstacles that appear on a real
route. They avoid start, goal and mapped geometry, and a detour stays open for a disc of the
robot's turning radius. Training maps only: test cases place their own (`test.unmapped_obstacles`).

```bash
pixi run train --preset navigation --set recipe.env.surprise_obstacles.probability=0.6
```

## Inspect or resume

```bash
pixi run dwarl runs list
pixi run dwarl runs watch
pixi run train --resume runs/<run> --max-iterations 5000
```

The resume budget is **5,000 total iterations**. Resume uses the saved recipe.
Use a new run to change the robot, observations or rewards.

## Start from another run's policy

```bash
pixi run train --preset navigation --set train.init_from=runs/<run>
```

A new run starts from the weights of another run's best checkpoint (its latest if none
was selected; `--set train.init_checkpoint=<iteration>` picks another), for example to train a changed reward on top of a policy that already
drives. Weights whose names and shapes match are copied; layers whose input changed,
the exploration noise, the optimizer and the iteration count start fresh. The robot and
the actor's observations should match the source run. Unlike `--resume`, the new run has
its own recipe and history.

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
