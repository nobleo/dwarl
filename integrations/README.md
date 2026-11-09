# Robot integrations for the Docker demo

Each directory here is a ROS package that DWARL can train, test and deploy for. It holds
a `dwarl_robot.yaml` profile and, for Gazebo testing, a `ros.sim_launch` adapter (see
[docs/integrate-robot.md](../docs/integrate-robot.md)). The Docker image builds every
package here, together with the robot's description sources, into `/opt/robots_ws`.

| Package | Robot | Description source |
| --- | --- | --- |
| `cindy_dwarl` | Nobleo Cindy: diff drive, rear omni-wheel, RPLidar A2 | `cindy_description` (Bitbucket, branch `feature/namespace`) |

The TurtleBot3 Waffle needs no package here. It is DWARL's builtin `diffbot` profile,
with the `tb3_sim.launch.py` adapter around `nav2_minimal_tb3_sim`.

## Description sources

Private description repositories are not cloned during the build. Instead,
`docker compose build` reads your local checkout and extracts a fixed git ref with
`git archive`, so your working tree and branch are not touched. If your checkout or ref
differ from the defaults, set these in `docker/.env`:

```bash
CINDY_DESCRIPTION_REPO=../../cindy_description   # relative to docker/
CINDY_DESCRIPTION_REF=origin/feature/namespace
```

Run `git -C <checkout> fetch` first so that the ref is current. If you have no checkout,
point `CINDY_DESCRIPTION_REPO` at an empty directory, for example `./empty`. The image
then builds without Cindy, and the demo offers only the other robots.

## Adding a robot

1. Create `integrations/<robot>_dwarl/` with `package.xml`, a `CMakeLists.txt` that
   installs `dwarl_robot.yaml` and `launch/`, the profile, and the Gazebo adapter launch
   file. `cindy_dwarl` is a working template.
2. If the description lives in another repository, add a build context and an extraction
   step to `docker/Dockerfile` and `docker/compose.yaml`, following the Cindy ones. Packages
   released as Debian packages can be installed with `apt-get` instead.
3. Add the robot to `demo/dwarl_demo/robots.yaml`, or to `work/robots.yaml` for local use only.
4. Rebuild with `docker compose build` and check the model with
   `docker compose run --rm shell dwarl robot check <robot>_dwarl`.

## Cindy notes

- The geometry comes unchanged from `cindy_description`. Its URDF has no rear omni-wheel,
  so `cindy_gz.urdf.xacro` adds a frictionless caster sphere for Gazebo. MuJoCo training
  uses an equivalent `collision.supports` sphere.
- The ros2_control block is defined here rather than taken from cindy's macro. That macro
  limits wheel velocity commands to ±1 rad/s, about 0.05 m/s, far below cindy's 0.26 m/s.
- The limits follow cindy's Nav2 velocity smoother (0.26 m/s, 1.0 rad/s), with
  conservative accelerations. The footprint is the convex hull of cindy's Nav2 footprint.
- `config/baselines.yaml` holds cindy's own RPP tuning (production, `main`) and DWB tuning
  for the controller comparison. Velocity and acceleration limits stay matched to the
  trained policy.
