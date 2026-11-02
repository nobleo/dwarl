# Troubleshooting

Start with the command that matches the problem:

```bash
pixi run dwarl robot check             # model and geometry
pixi run dwarl doctor                  # MuJoCo/inference environment
pixi run dwarl doctor --sim gazebo --live  # running simulator interfaces
pixi run dwarl status                  # selected robot, run and evidence
```

| Problem | Next action |
| --- | --- |
| No robot selected | Run `pixi run setup`, or pass the integration path. |
| Package or mesh not found | Select the correct workspace in setup; verify the profile URI. |
| Joint/link not found | Rerun `robot init` and choose a name from the expanded URDF. |
| Model falls or tips | Inspect inertials, wheel collisions and support/caster geometry in `robot view`. |
| First run takes time | Read the dataset-generation stage; matching later runs reuse the cache. |
| `best` unavailable | Complete periodic validation. Use `latest` for pipeline checks; manual tests do not update best selection. |
| Gazebo startup fails | Run the live doctor; inspect its printed `launch.log` for missing interfaces. |
| Gazebo is slow | Check real-time factor, machine load and doctor findings; compare runs under matched conditions. |
| Incorrect spawn position | Check world-coordinate versus zero-origin odometry and duplicate transforms. |
| Controller stops on scans | Check topic, timestamps, filtering and base-to-sensor TF. |
| Nav2 runs but robot stays still | Check Twist/TwistStamped, command topic, mux and collision-monitor state. |
| Deployment incompatible | The profile or referenced model differs from the frozen run; use the matching integration or train a new run. |
| Wizard stopped midway | Rerun `robot init` with the same destination; its draft is retained. |

Use `--verbose` for detailed diagnostics. Builds save full logs under `.dwarl/logs/`;
tests save stack logs in their case directories. `--json` gives structured results,
while `--plain` disables terminal styling and animation.
