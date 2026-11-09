# DWARL

**Train, test and deploy a neural local planner for your differential-drive robot.**
Train in MuJoCo, evaluate exact checkpoints in MuJoCo or Gazebo, and install an
ONNX policy bundle for the ROS 2 Nav2 controller plugin.

[Documentation](https://nobleo.github.io/dwarl/) ·
[Quickstart](https://nobleo.github.io/dwarl/quickstart/) ·
[Integrate your robot](https://nobleo.github.io/dwarl/integrate-robot/)

## Get started

Install [pixi](https://pixi.sh), then:

```bash
git clone https://github.com/nobleo/dwarl.git
cd dwarl
pixi run setup
```

Setup lets you try the included example, integrate your robot, or continue an
existing integration. It remembers your selection, helps create a robot profile,
and checks the requested environment. Python, ROS 2 Jazzy and Gazebo Harmonic come
from pixi. No manual sourcing is needed. Supported environment: **Linux x86-64**.

**Using a coding agent?** Follow [agent-assisted setup](https://nobleo.github.io/dwarl/agent-setup/)
for a copy-paste prompt and the included portable robot-setup skill.

### Docker

To keep ROS 2 Jazzy, Gazebo Harmonic and Nav2 off your host, use the image in
[docker/](docker/README.md). It also contains an end-to-end demo: pick a robot (a
TurtleBot3, or your own), train, compare with Nav2's MPPI, DWB and Regulated Pure Pursuit
in Gazebo, including obstacles that are not on the map, deploy, and watch the policy drive
under Nav2:

```bash
cd docker && ./configure-host.sh && docker compose build
docker compose run --rm demo
```

## The workflow

```bash
pixi run train --preset smoke       # small pipeline check
pixi run train --preset navigation  # starter recipe with periodic validation
pixi run test --suite validation    # evaluate the selected run
pixi run dwarl report --open        # inspect individual outcomes
pixi run deploy                    # install a compatible bundle
```

Each command reports its selected inputs and a concrete next step. Use
`pixi run dwarl status` for an overview or `pixi run dwarl next` for guidance.
Explicit robot paths, runs and checkpoints remain available through `--help`.

DWARL replaces the local controller, not the complete navigation stack. You bring
the robot description, sensing, drivers and localization. Integrations remain in
your robot repository; DWARL stays robot-neutral.

DWARL is experimental and includes no pretrained navigation-quality policy.
A smoke run checks the pipeline. Choose policies on validation data and use
held-out tests for final evaluation.

## Development

```bash
pixi run pytest
pixi run lint
pixi run docs
```

See [contributing](https://nobleo.github.io/dwarl/contributing/) for environment
variants, ROS checks and documentation preview.

## License

[Apache License 2.0](LICENSE) — Copyright 2026 Nobleo Technology. See [NOTICE](NOTICE).
