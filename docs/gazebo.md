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

See the [full adapter contract](manual-integration.md#6-add-gazebo-testing) and
[runtime measurement conventions](results.md#measurement-conventions).
