# Run the example in Docker

From a Linux x86-64 checkout with Docker and a local X11 or XWayland display:

```bash
./docker/run demo
```

The first run builds the image, then opens MuJoCo. Later runs reuse the local
image; use `./docker/run build` to rebuild after checkout changes.
Press Ctrl+C when finished. To run the printed test or
report command, replace `pixi run test` or `pixi run dwarl report` with
`./docker/run test` or `./docker/run report` and keep the arguments.
Files under `/work` in the container are in `work/` on the host; set `DWARL_WORK`
to use another directory. Open generated HTML reports on the host.

For an NVIDIA-enabled Docker host, use `./docker/run --gpu demo`. Headless CLI
commands need no display; for example, `./docker/run status`.

Docker is for the built-in example. For your own ROS workspace or deployment,
use the [native Pixi integration guide](../web/docs/integrate-robot.md).
