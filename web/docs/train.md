# Train a policy

**Before starting:** select and check a robot with [setup](integrate-robot.md).
Configure its Gazebo adapter first if you intend to use Gazebo with this run.

## Start with validation enabled

```bash
pixi run train --preset navigation
```

Use `--num-envs` to fit training to your machine, `--max-iterations` to set the
total budget, and `--set train.validation_interval=…` to schedule checks.
For example, preview a smaller run without starting it:

```bash
pixi run train --preset navigation --num-envs 8 --print-config
```

Training reuses matching maps or generates them on first use. To prepare them ahead of time, run [datagen](datagen.md).

The CLI shows model preparation, training progress and
the completed checkpoint. The run contains a copy of the robot assets and records
the dataset path and index hash. Keep the referenced dataset if you move the run.

## Obstacles that are not on the map

`recipe.env.surprise_obstacles` adds random boxes and cylinders to the reference path of a
share of training episodes. The map and the path stay as they
were, so the robot only sees them in its lidar, like obstacles that appear on a real
route. They avoid start, goal and mapped geometry. By default their diameter or
longer box side is 50–90% of the robot's footprint width before curriculum scaling.
An explicit `size: [min, max]` still specifies radius or box half-extent in metres.
Placement requires a short, footprint-clear sidestep back onto the reference path.
Test cases place their own (`test.unmapped_obstacles`).

```bash
pixi run train --preset navigation --set recipe.env.surprise_obstacles.probability=0.6
```

With `recipe.env.surprise_obstacles.curriculum=true`, training starts on clean
routes and adjusts obstacle difficulty using success and collision rates. Set
`probability` for the final share of episodes that *attempt* placement, `count`
and `size` for obstacle ranges, and `margin` for detour clearance. Placement can
fail. Track `Curriculum/obstacle_probability` in TensorBoard.

Set `recipe.env.surprise_obstacles.probability=0` for clean routes, or
`recipe.env.surprise_obstacles.curriculum=false` to keep the probability constant.

Use `recipe.agent.algorithm.entropy_coef` to tune the exploration bonus.

When an obstacle blocks the reference path, training reduces the weight of
cross-track and heading rewards so the policy can leave the path. Other terms,
including proximity, collision and angular effort, still apply.

To penalize steering away from or overshooting the lookahead target, tune
`recipe.rewards.dewobble.weight` and `recipe.rewards.dewobble.params.horizon`
in a new run; a zero weight disables the term. To simulate monitor stops, use
`recipe.env.safety_layer.enabled`. The training layer does not accept custom
stop-zone margins, latency or point thresholds; verify any separately tuned
deployment monitor on the target stack.

Only the critic sees the avoidance level. The actor's observations do not change, so the
exported policy and the Nav2 plugin are the same as for path following. Validation of such a
recipe places unmapped obstacles in its cases, so the selected checkpoint reflects both skills.
Set `test.unmapped_obstacles` to request obstacles per case in MuJoCo or Gazebo. Use `--set test.unmapped_obstacles=0`
for a clean baseline, and check actual placement counts before claiming avoidance.

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
was selected; `--set train.init_checkpoint=<iteration>` picks another), for example to train a
changed reward on top of a policy that already drives. Weights whose names and shapes match are
copied, including the learned exploration noise; layers whose input changed, the optimizer and
the iteration count start fresh. The robot and
the actor's observations should match the source run. Unlike `--resume`, the new run has
its own recipe and history.

Add `--detached` for background training or `--show` for the simulator viewer.
`--show` trains in one unsharded world, which is much slower. To look at a headless run
instead, run `pixi run dwarl runs view` from a second terminal: a viewer shows the first
simulation shard while training continues. Repeat the command or close the window to
return to full speed. If no window can be opened (no display, or no X11 access), the
training prints `live view unavailable` and continues headless.
[Monitoring and resume](monitor.md) covers logs, TensorBoard and interruption.

??? info "Presets and configuration"

    Use `smoke` to check the pipeline or `navigation` to train with periodic
    validation. Presets are recorded in the run.

    Inspect the resolved settings with `--print-config` or inputs with `--dry-run`.
    [Configuration](configuration.md) explains YAML, overrides and early stopping.

**Next:** [evaluate a checkpoint](test.md). The CLI prints a pinned test command.
