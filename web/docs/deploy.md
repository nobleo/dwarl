# Deploy to Nav2

**Goal:** install a compatible ONNX bundle and launch the DWARL controller.
Your robot must already provide drivers, filtered scan, odometry, TF, map and localization.

## Install a checkpoint

```bash
pixi run deploy --run runs/<run> --checkpoint <iteration>
```

DWARL checks the current integration against the frozen training model, prepares
the ONNX bundle and updates `selected_policy.yaml`. It reports recorded evaluation
evidence; installation does not certify navigation quality.

The bundle includes the model, deployment contract, assets and provenance.
**Keep the whole bundle together.** New weights do not require recompiling the controller.

## Launch on this computer

After starting the robot interfaces and localization, use the complete command
printed by deploy. For a wizard-generated integration it looks like:

```bash
pixi run ros2 launch my_robot_dwarl navigation.launch.py \
  integration:=/absolute/path/to/my_robot_dwarl
```

The wrapper reads the selected policy. Set `use_sim_time:=true` for a simulator.
Open RViz and send a Nav2 goal. DWARL's standard navigation launch supplies Nav2 and
a collision monitor; it expects the drivers, map and localization to be running.

## Use another robot computer

Copy the **complete bundle** to the target. On a supported Linux x86-64 target,
clone DWARL and run its dedicated runtime environment:

```bash
pixi run -e runtime runtime-launch model_dir:=/absolute/path/to/bundle use_sim_time:=false
```

This builds the native packages in a separate directory and activates them automatically.
The runtime environment contains Nav2 and inference dependencies, without PyTorch
or MuJoCo. The lockfile currently targets Linux x86-64; other architectures need
separate packaging and verification.

??? info "Already have a Nav2 stack?"

    Load `dwarl_controller::DwarlController` in your controller server and pass
    the complete bundle as `model_dir`. Match the robot ID, control frequency,
    footprint, topics and command routing. See the
    [existing-stack configuration](manual-integration.md#already-have-your-own-nav2-stack).

**Need help?** [Troubleshooting](troubleshooting.md) covers incompatible bundles and motion interfaces.
