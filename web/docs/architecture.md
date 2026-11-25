---
hide:
  - toc
---

# How DWARL works

Nav2 plans a route. DWARL chooses the next velocity. Select **DWARL** to step inside the controller and see how it learns, or select another component to inspect its role.

<div id="theory-explorer"></div>

<details class="theory-notes" markdown="1">
<summary>Read the diagram as text</summary>

In Nav2, the global planner supplies a path. Lidar and motion state meet that path at DWARL. Its requested command passes through the collision monitor to the robot. In simulation, observation → actor → dynamic window → environment → next observation forms a loop. PPO uses the rollouts to update the actor. Only the trained actor and its deployment contract are needed for inference.

## Maps & routes { #routes }

A training episode needs a reachable goal as well as a world. The ROSCon examples use clutter, mazes, rooms and caves: a doorway tests clearance, while a winding corridor tests tracking over repeated turns. Inflating occupied cells for the robot footprint before planning keeps a point-sized planner from drawing impossible routes. Start–goal pairs are connected through free space and ranked by length, detour, narrowness and density.

A curriculum changes which pairs are sampled as success improves; easier routes remain useful practice. In the avoidance recipe, a second curriculum places obstacles **absent from the reference map**. These are different kinds of difficulty: even a short straight route can require a local detour. [Generate datasets](datagen.md).

## MuJoCo environment { #environment }

The simulator applies the velocity command to a model of the robot, advances its dynamics and samples the next scan and motion state. Commanded speed and measured speed can differ. Feeding measured motion back matters: the next decision starts from what the robot actually did.

The environment can evaluate footprint contact, progress along the known route and safety stops. It can use this information to score a step or end an episode without giving the actor a map of hidden obstacles. Many independent robots can collect these transitions in parallel. [Train in simulation](train.md).

## Rollouts & reward { #reward }

A rollout records a sequence of observations, actions, rewards and episode endings. Moving along the route earns progress; reaching the goal yields a terminal reward. Collision and getting stuck penalize the episode. Time costs discourage simply waiting; clearance, reversing and angular-acceleration terms shape *how* the robot moves.

These objectives can compete. If a box blocks the centerline, punishing every centimeter of cross-track error encourages pushing straight ahead. Relaxing tracking penalties near a blockage lets the robot trade a short deviation for later progress. A stop penalty discourages relying on the collision monitor to brake. Reward weights describe training incentives, not runtime guard thresholds. [Inspect configuration](configuration.md).

## Critic & PPO { #learning }

The **actor** chooses an action. The **critic** estimates the expected discounted return from the current state. In asymmetric actor–critic training the critic can see privileged simulator state: it can make a better estimate without teaching the actor to depend on signals missing on the robot.

Proximal policy optimization (PPO) compares rollout returns with the critic’s predictions. Actions that worked better than expected become more likely; clipping limits how far an update can move the policy on the sampled actions. Discounting determines how much later progress matters, and advantage estimation balances noisy long-term outcomes against the critic’s shorter-term predictions. This is why a brief detour can be reinforced even when its first step earns little. The updated actor collects another rollout; neither PPO nor the critic is needed in Nav2.

## Observation { #observation }

The actor sees a **local** navigation problem. In the ROSCon example, the lidar input contains 180 rays at each of three instants. The navigation recipe samples the current scan and those from two and five control ticks ago. Four lookahead points describe the desired route relative to the robot. Measured linear and angular velocity plus the previous action provide motion context. The path can suggest turning left even if the scan alone looks symmetric.

Ray order, units, frames, history order and scaling give those numbers meaning. A runtime vector with the right length but reversed rays would describe a different scene. The deployment contract keeps preprocessing aligned with training. The ROSCon dimensions illustrate that model; use the selected bundle for the installed policy.

## Actor network { #actor }

A one-dimensional convolution slides across adjacent lidar rays, learning local patterns such as an obstacle edge or the opening between two surfaces. In the slide model, successive layers see spans of 7, 15 and 23 original rays. The scan encoder produces 128 features rather than treating every ray as a separate steering rule.

A multilayer perceptron combines those features with path and motion state (hidden widths 512, 256 and 128 in the slide). Its two outputs request changes in forward and turning speed. During training, sampling actions explores alternatives; the deployed actor runs inference with fixed weights. It produces a local decision, not a new global path.

## Dynamic window { #window }

The actor requests a change, not an unrestricted velocity. In window mode each normalized action is clipped to [−1, 1], scaled by the acceleration limit times the control period, then added to the **previous command**. The resulting linear speed **v** and angular speed **ω** are clamped to the robot’s velocity limits. Measured speed is an observation, not the starting point of this update.

At the slide’s 10 Hz control rate, a decision covers 0.1 seconds. A fast-moving robot cannot instantly reverse, even if the scan changes abruptly. Acceleration feasibility does not imply obstacle clearance: a reachable command can still be unsafe, which is why the command guard is a separate box.

## Checkpoint selection { #validation }

Training produces a succession of actors. Use matched validation cases to compare checkpoints: did the change improve goal completion, reduce collisions, or just make the robot stop sooner? Averages can hide a regression on narrow routes; inspect per-case outcomes and trajectories. Reserve the held-out test set for the final assessment. Export freezes a decision about which actor to use; it does not improve that actor. [Read evaluation results](results.md).

## ONNX bundle { #bundle }

ONNX carries the actor’s inference graph and weights across the Python-to-native boundary. The bundle also supplies the deployment contract, assets and provenance. Weights alone cannot tell a controller which robot, scan convention or command limits they were trained with. Install the complete compatible bundle. No simulator, optimizer or reward calculation crosses this boundary. [Deploy a checkpoint](deploy.md).

## Global planner { #planner }

Given a goal, global costmap and localized robot pose, Nav2’s planner supplies a reference path through the environment. It reasons over a larger area than the actor can see with lidar. Changing the global route is a planner decision; steering around a temporary obstruction can be a local one.

## Global path { #path }

The path is a sequence of poses, not a timetable of wheel commands. DWARL projects the robot onto a path segment, then samples points ahead along the route. The navigation recipe looks 0.5, 1, 2 and 3 metres ahead. Near points describe the next bend; farther ones tell the actor which direction to resume after a detour.

## Lidar + state { #sensors }

The controller uses the laser-to-base transform to place scan returns in the trained observation frame and bins them by angle. Odometry provides actual forward and turning speed. A scan too old to use is a controller failure, not evidence of a clear hallway. These inputs let the policy distinguish an open route from a blocked one, or a moving robot from a stopped one.

## DWARL controller { #controller }

The native Nav2 controller brings the two inputs together: the reference path in the robot’s frame, and current scan and motion feedback. It builds the actor observation, runs the exported network and converts the result through the dynamic window into a requested velocity. The plugin does not train online. [Connect an existing Nav2 stack](manual-integration.md#already-have-your-own-nav2-stack).

## Collision monitor { #guard }

With the standard safety layer enabled, the controller publishes `cmd_vel_raw`. The monitor selects a stop polygon for that motion and checks live scan returns against it. Enough returns inside the zone trigger a stop; otherwise the command passes to the robot’s configured velocity topic. The zones derive from footprint and braking distance: a faster command needs more space. This check is independent of the actor.

A guarded stop buys time; it does not find a route around the obstruction. Routing the controller output directly to the drive would skip this stage. [Check command routing](deploy.md).

## Robot & feedback { #robot }

The drive executes the guarded command. Lidar observes the resulting scene; odometry reports actual motion, including a stop imposed downstream of the actor. Transforms relate the scan, robot and path frames. On the next tick these measurements close the loop: the actor reacts to the new situation rather than replaying a stored sequence of steering commands. [Robot interface requirements](integrate-robot.md).

## Recovery & replanning { #recovery }

A stop is not a new path. In the standard safety-retry tree a monitor stop can fail `FollowPath`; Nav2 backs up and retries the controller. If retries are exhausted, the tree can replan or recover. A new path changes the lookaheads, not the actor’s weights. The actor proposes motion, the guard may stop it, and Nav2 manages the task.

</details>
