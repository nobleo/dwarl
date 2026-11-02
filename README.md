# DWARL

Train a differential-drive local planner, test an exact checkpoint, and install it into a robot integration.
The Python package and command are `dwarl`.

DWARL is experimental. No pretrained navigation policy is included; evaluate trained
policies for your robot and operating environment before deployment.

## Documentation

**Documentation site:** [nobleo.github.io/dwarl](https://nobleo.github.io/dwarl/)
(built from `docs/` on every main push).

Start with **[Your robot: from model to Nav2](docs/integrate-robot.md)**:
define a robot profile, train, test, export the ONNX policy and run it on your
robot with Nav2. Includes a profile template and Gazebo adapter requirements.
See the [documentation overview](docs/index.md), [configuration](docs/configuration.md),
[test results](docs/results.md), and [ROS setup](docs/ros.md) for the full workflow.

The searchable site and command reference are generated on every main push.
You can also download the `documentation` artifact from
[GitHub Actions](https://github.com/nobleo/dwarl/actions/workflows/docs.yml),
or [preview the site locally](docs/documentation.md).

## Install

Install [pixi](https://pixi.sh) once (`curl -fsSL https://pixi.sh/install.sh | bash`). It installs
Python, the training stack, ROS 2 Jazzy, Nav2 and Gazebo into `.pixi/`; nothing needs to be sourced:

```sh
git clone https://github.com/nobleo/dwarl.git
cd dwarl
pixi run setup            # asks for your robot repository, builds ROS + robot, checks Gazebo
pixi run train /path/to/robot-repository
pixi run test --sim both
pixi run deploy /path/to/robot-repository
```

`pixi run setup` remembers its answers in `.dwarl/setup.env`; rerun it to change them.
Every `dwarl ...` command below also works inside `pixi shell` (or as `pixi run dwarl ...`).
For a slim training-only machine without ROS, use `pixi install -e train`.

## Workflow

```sh
dwarl train /path/to/robot-repository
dwarl test --case doorway
dwarl test --sim gazebo --show
dwarl test --sim both
dwarl deploy /path/to/robot-repository
```

Training prepares robot-sized datasets automatically. Test and deploy select the
latest **completed** checkpoint of the latest run once, print its hash, and prepare
an immutable ONNX bundle automatically. `--checkpoint best` is explicit and requires
completed validation evidence. A fresh one-iteration smoke policy is not a usable planner.

Intermediate checks and early stopping are off by default. Enable validation with
`--set train.validation_interval=200`, benchmarks with `--set train.benchmark_interval=500`,
and validation-based stopping with `--set train.early_stop.enabled=true`.
See [configuration](docs/configuration.md#optional-intermediate-testing-and-early-stopping).

For a standalone MuJoCo smoke test, use the included `diffbot`:

```sh
dwarl train diffbot --num-envs 2 --max-iterations 1 \
  --set train.validation_interval=0 \
  --set train.data.train_maps=1 --set train.data.validation_maps=1 \
  --set train.data.test_maps=1 --set train.data.pairs_per_map=3
dwarl test --case doorway --set test.max_cases=1
```

See [configuration](docs/configuration.md), [results](docs/results.md), and
[native ROS setup](docs/ros.md).

## License

[Apache License 2.0](LICENSE) — Copyright 2026 Nobleo Technology. See [NOTICE](NOTICE).
