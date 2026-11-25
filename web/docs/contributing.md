# Development

Use pixi for every environment:

| Environment | Purpose |
| --- | --- |
| `default` | Training, ROS 2 Jazzy, Gazebo Harmonic, and dashboard development |
| `train` | Training and Python tests without ROS or Node.js |
| `runtime` | Native Nav2 controller and inference; no training stack |
| `docs` | Documentation tooling only |

The lockfile targets Linux x86-64. Select an environment with `pixi run -e <name>`.
Pixi activates the ROS overlays; do not source workspaces by hand. Commit `pixi.lock`
with dependency changes.

Keep Python package tests in `dwarl/tests/` and workspace integration tests in
`tests/integration/` (for example, Python policy export versus the native ROS
controller). `pixi run pytest` runs both from the repository root.
Component scripts likewise belong in `ros/scripts/` or `web/scripts/`; the root `scripts/` holds Pixi workspace activation. Browser tests
live in `web/tests/`, and each ROS package owns its native tests under `test/`.

```bash
pixi run pytest
pixi run lint
pixi run -e docs docs
git diff --check
```

For dashboard changes, `pixi run web-test` builds the frontend and the bundled docs
(served at `/docs/`) before running Playwright. On a new machine, install Chromium once with
`pixi run web-build` followed by `pixi run npm --prefix web exec -- playwright install chromium`.

For native ROS changes, `ros-test` builds before testing. Wait for any live
worker using the controller library to finish before rebuilding it.

```bash
pixi run ros-test
```

Integrations and private robot data belong in the robot repository. Public fixtures
use the included `diffbot` or `dwarl/src/dwarl/config/fixtures/testbot/`.

Human CLI output uses `dwarl.cli.ui` on stderr. Structured output uses stdout.
Keep the parser lightweight, and derive documentation from command/schema metadata.

[Maintain and preview documentation](documentation.md) · [ROS build details](ros.md)
