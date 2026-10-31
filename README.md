# DWARL

Train a differential-drive local planner, test an exact checkpoint, and install it into a robot integration.
The Python package and command are `dwarl`.

**[First-time Cody walkthrough](docs/cody.md)** · **[Documentation overview](docs/index.md)**

The searchable documentation site is generated on every main push; see
[preview and Pages setup](docs/documentation.md). The walkthrough includes
TensorBoard, resume, paired simulator testing and deployment.

DWARL is experimental. No pretrained navigation policy is included; evaluate trained
policies for your robot and operating environment before deployment.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/) are recommended:

```sh
git clone https://github.com/nobleo/dwarl.git
cd dwarl
uv sync --group dev
source .venv/bin/activate
dwarl --help
```

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

[MIT](LICENSE) — Copyright (c) 2026 Nobleo Technology.
