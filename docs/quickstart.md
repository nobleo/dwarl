# Try the example robot

**Goal:** exercise training, checkpoint export and inference using the included
`diffbot`. You need Linux x86-64 and [pixi](https://pixi.sh).

## 1. Select the example

```bash
git clone https://github.com/nobleo/dwarl.git
cd dwarl
pixi run setup --example
```

Pixi installs the environment on first use. Setup selects the example robot;
this path needs no robot repository or ROS build.

## 2. Run a small training check

```bash
pixi run train --preset smoke
```

Expect a tiny dataset, two training iterations and a completed checkpoint.
The command prints the exact run-specific test command. You can also use:

```bash
pixi run test --case doorway --set test.max_cases=1
```

A smoke policy is deliberately untrained. Navigation failure (exit code `1`) is
expected; an infrastructure failure (exit code `2`) needs investigation.

## 3. See what happened

```bash
pixi run dwarl report --open
pixi run dwarl next
```

The report includes individual trajectories and motion traces. On a headless
machine, the command prints the local report path.

**Next:** [integrate your robot](integrate-robot.md), or [train a navigation policy](train.md).
