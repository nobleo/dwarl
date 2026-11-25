# Add Gazebo evaluation

**Goal:** connect an existing robot simulator to DWARL's managed tests.
Complete this before training a run that needs Gazebo.

## Generate a wrapper

```bash
pixi run dwarl robot init ../my_robot/src/my_robot_dwarl --gazebo
```

Choose your robot-only simulator package and launch file. Map DWARL's
`world`, `x`, `y`, `yaw` and `gui` arguments to the names that launch accepts.
The generated wrapper sets `ros.sim_launch` in the profile.

The simulator must start the robot, sensors, drive controllers and bridges.
DWARL supplies the navigation stack and case-specific map during evaluation.

## Check real behavior

```bash
pixi run dwarl doctor --sim gazebo --live
```

The doctor builds required ROS packages, launches an isolated test world at a
nonzero pose, and checks:

- Advancing clock, scan and odometry.
- Odometry-to-base and base-to-sensor transforms.
- The expected command subscription and message type.
- Ground-truth entity `robot` at the requested pose in world `default`.
- Odometry starting at zero at spawn.

This check sends no motion commands and uses no trained policy. Full logs and
findings are saved under `.dwarl/checks/`. A generated wrapper alone does not
establish that these conditions hold.

## Evaluate a checkpoint

```bash
pixi run test --sim gazebo --suite validation
```

Use `--sim both` to evaluate matching cases in MuJoCo and Gazebo. Gazebo cases run
sequentially with isolated domains and partitions.

## Choose who supplies localization

**By default, Gazebo tests use perfect localization:** DWARL combines the
simulator's model pose with odometry TF at the same timestamp to publish
`map → odom`. This isolates controller behavior from localization drift. The
adapter still supplies `odom → base`. If the Gazebo model origin differs from
`ros.base_frame`, set `ros.gazebo_model_frame` in the robot profile to a rigid
TF frame at that origin and publish its fixed transform to the base.

**To test your own localizer**, run:

```bash
pixi run test --sim gazebo --external-localization
```

This disables DWARL's `map → odom` publisher; it does not start a localizer or
consume a pose topic directly. The simulator adapter receives the launch argument
`external_localization:=true`. Use it to start your localizer in the case's ROS
domain, with simulation time and the case map on `/map`. Initialize it for the
requested spawn pose. It must publish the profile's `map → odom` TF; if that
transform never arrives the test times out rather than falling back to truth.
Only one node should own that transform. Direct `bench.launch.py` users can
pass `external_localization:=true` too.

Both modes score trajectories against Gazebo truth. Compare controllers using
the same localization mode; the frozen request and episode diagnostics record it.

The safety monitor activates before command-producing servers, and tests wait for
both lifecycle managers to finish establishing bonds before sending a goal.
Nav2 monitors lifecycle heartbeats. A heartbeat shutdown is an
infrastructure failure, not evidence of controller quality: inspect `launch.log`
and measure simulator speed with `--profile-compute`.

See the [full adapter contract](manual-integration.md#6-add-gazebo-testing) and
[runtime measurement conventions](results.md#measurement-conventions).
