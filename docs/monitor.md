# Monitor and resume

## Inspect progress

```bash
pixi run dwarl runs list
pixi run dwarl runs show
pixi run dwarl runs watch
```

Pass a run directory after `show` or `watch` to select a different run. Progress
includes iteration, elapsed time and a measured ETA. Raw training output is saved
in `runs/<run>/training.log`; TensorBoard events live under `checkpoints/`.

```bash
pixi run tensorboard --logdir runs --host localhost --port 6006
```

Open <http://localhost:6006>. Reward and losses help diagnose learning; validation
scores establish navigation performance.

## Background execution

```bash
pixi run train --preset navigation --detached
```

The command prints a process log path. Once model/data preparation completes, the
run is available through `runs list` and `runs watch`.

## Stop and continue

```bash
pixi run dwarl stop runs/<run>
pixi run train --resume runs/<run> --max-iterations 5000
```

The budget is total iterations, not additional iterations. Resume restores the saved
recipe and completed checkpoint; simulator episodes restart. Ctrl+C also retains
completed checkpoints and available partial test evidence.

[Early stopping and detailed metrics](configuration.md#optional-intermediate-testing-and-early-stopping)
