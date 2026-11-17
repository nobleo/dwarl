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
share of training episodes (`probability`, a final target of 0.6 by default). The map and the path stay as they
were, so the robot only sees them in its lidar, like obstacles that appear on a real
route. They avoid start, goal and mapped geometry, and a detour stays open for a disc of the
robot's turning radius. Training maps only: test cases place their own (`test.unmapped_obstacles`).

```bash
pixi run train --preset navigation --set recipe.env.surprise_obstacles.probability=0.6
```

### Default avoidance curriculum

Plain training and the navigation preset now start with clean routes, then introduce
obstacles through five stages. Each stage raises the placement probability by 12 percentage
points, reaching 60%. The maximum count grows from one to three; maximum obstacle radius/half
extent grows from 0.16 m to 0.4 m. Extra detour margin decreases from 0.26 m to 0.1 m.
These are placement targets; rejected placements may yield fewer or no obstacles.

The first promotion requires 200 clean episodes with at least 75% success and at most 5%
collisions. Later decisions require 200 clean **and** 200 actually obstacle-containing
episodes at the current stage, each meeting those thresholds. Success at or below 30% or
collisions above 10% in either group demotes one stage; otherwise the stage holds. Windows
clear after a decision, and episodes started at other stages are ignored. Route-difficulty
progression remains separate. Clean episodes remain in the mix at every stage.

The stage and partial outcome windows are checkpointed. TensorBoard reports
`Curriculum/avoidance_stage` and `Curriculum/obstacle_probability`.
These thresholds are starting heuristics, not a validated performance improvement.

Set `recipe.env.surprise_obstacles.probability=0` for clean-only training, or
`recipe.env.surprise_obstacles.curriculum=false` for a constant obstacle probability.
The smoke preset disables surprise obstacles.

### Compatibility preset: `--preset avoidance`

```bash
pixi run train --preset avoidance
pixi run train --preset avoidance --set train.init_from=runs/<path-following run>
```

The path rewards (cross-track error and heading along the path) punish every deviation, so a
path-following policy collides with or stops at an obstacle on its path. The avoidance
curriculum eventually attempts obstacles on the path in 60 % of episodes and computes an avoidance level
between 0 and 1 from privileged geometry: how close the path within the next few metres
(`horizon`, `horizon_time`) passes to such an obstacle (`clear`, `ramp`), weighted up as the robot
gets nearer (`near`). Cross-track and heading rewards are scaled by one minus that level, so
while the path is blocked a detour costs only progress, and the proximity and collision
penalties keep their full weight. Behind the obstacle the level decays over `release_time`
(a finite, strictly positive number of seconds)
and the policy is rewarded for rejoining the path.

Two more terms keep the detours usable on a robot. `safety_zone` costs as much as a short
collision whenever the collision monitor of the deployed stack would stop the robot. Its STOP
zone follows the commanded velocity: the footprint plus 3.5 cm, stretched forward by the braking
distance when driving, and an octagon around the footprint when turning on the spot. On the robot
such a stop starts recoveries, so passing an obstacle with millimetres to spare is not a
success. `angular_effort` penalises angular acceleration; with the `window` action mode the
action is the command change, so `action_rate` does not see weaving. The preset also lowers
the entropy bonus, so the exploration noise does not grow into bang-bang steering.

The default `dewobble` term explicitly penalises steering that conflicts with the
desired heading. It uses the bearing to the main lookahead target before the action
and the limited angular command before safety overrides. With heading error `e` and
`horizon=0.5` seconds, useful angular commands lie between zero and `e / horizon`.
The squared distance outside this interval, normalised by the angular speed limit,
is penalised with weight 2.0 (the reward manager also multiplies by the control timestep).
Thus continued turning when aligned, wrong-way turns and excessive steering cost
reward; braking reduces the cost. The lookahead bearing permits useful turns on
bends and when returning from a lateral offset. The penalty relaxes during blocked-path
detours. This is a heading-aware shaping heuristic, not a guarantee of less wobble.
Tune `recipe.rewards.dewobble.weight` and `recipe.rewards.dewobble.params.horizon`
in a new run; set the weight to zero to disable it.

The default recipe also enables `recipe.env.safety_layer`: training then stops the robot like the
monitor does, and the command restarts from rest with cleared previous-action and lidar-history
observations, as it does after Nav2's recovery. The policy
learns that a pose too close to an obstacle leaves it standing, for instance unable to turn
away, instead of only paying for it. MuJoCo tests of such a run apply the same layer.
Only the deployed defaults are supported for `margin` (0.035 m), `latency` (0.15 s),
and `min_points` (2). Custom training values are rejected because they are not exported
to the robot's collision monitor. Separate deployment tuning can change that monitor,
so verify its settings before attributing results to training/deployment parity.

Only the critic sees the avoidance level. The actor's observations do not change, so the
exported policy and the Nav2 plugin are the same as for path following. Validation of such a
recipe places unmapped obstacles in its cases, so the selected checkpoint reflects both skills.
Check the result with `dwarl test`, in MuJoCo and in Gazebo: testing requests up to
three unmapped obstacles per case by default. Use `--set test.unmapped_obstacles=0`
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

    `smoke` is a two-iteration pipeline check; `navigation` explicitly enables
    periodic validation. Plain `pixi run train` retains the normal defaults with
    intermediate checks disabled. Presets are recorded in the run.

    Inspect the resolved settings with `--print-config` or inputs with `--dry-run`.
    [Configuration](configuration.md) explains YAML, overrides and early stopping.

**Next:** [evaluate a checkpoint](test.md). The CLI prints a pinned test command.
