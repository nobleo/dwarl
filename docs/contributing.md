# Development

Use pixi for every environment:

| Environment | Purpose |
| --- | --- |
| `default` | Training, ROS 2 Jazzy and Gazebo Harmonic |
| `train` | Training-only jobs without ROS |
| `runtime` | Native Nav2 controller and inference; no training stack |
| `docs` | Documentation tooling only |

The lockfile targets Linux x86-64. Select an environment with `pixi run -e <name>`.

```bash
pixi run pytest
pixi run lint
pixi run docs
```

For native ROS changes:

```bash
pixi run build
pixi run ros-test
```

Integrations and private robot data belong in the robot repository. Public fixtures
use the included `diffbot` or `tests/robots/testbot`.

Human CLI output uses a shared renderer on stderr. Structured output uses stdout.
Keep the parser lightweight, and derive documentation from command/schema metadata.

[Maintain and preview documentation](documentation.md) · [ROS build details](ros.md)
