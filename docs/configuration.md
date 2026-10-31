# Configuration

Precedence: built-in typed defaults → `./dwarl.yaml` → command-line assignments.
`--config experiment.yaml` replaces the workspace file. YAML paths are relative to
that file; command-line paths are relative to the invoking directory. Duplicate CLI
assignments are errors. `--set` uses the actual schema key.

```yaml
schema: dwarl/1
robot: /path/to/robot-repository
train:
  name: cody
  num_envs: 64
  seed: 42
  device: auto
  max_iterations: 3000
test:
  checkpoint: latest
  suite: benchmark
  sim: mujoco
  show: false
deploy:
  checkpoint: latest
```

Inspect settings with `dwarl test --print-config`. Preview selection with
`dwarl test --dry-run`. Use `--run runs/<name>` to select a run explicitly.

`dwarl train --resume runs/<name>` restores the saved recipe. Device, viewer and
training budget may change. Simulator episodes restart; RNG/sampler bookkeeping is
restored, but continuation is not bitwise identical to uninterrupted physics.

Automatic data caches include robot clearance, generation parameters, seed and
generator version. An explicit `train.data.path` is checked for footprint compatibility;
`train.data.allow_custom=true` acknowledges an intentional custom dataset.

Periodic validation uses the validation split only. Best means maximum success rate
over a completed, identical validation workload; ties keep the earliest selection.
The held-out `test` suite is never used for checkpoint selection.

## Monitor training

TensorBoard is included in the Python environment. From your DWARL working
directory, run this in a second activated terminal:

```bash
tensorboard --logdir runs --host localhost --port 6006
```

Open <http://localhost:6006>. To focus on one run, use
`--logdir runs/<run>/checkpoints`. Events appear once training begins, after
dataset generation. Inspect reward, episode length, losses and throughput;
use scored validation and test reports to assess navigation quality.

Add `--show` to watch training or `--detached` to run it in the background.
List runs with `dwarl runs list`, stop with `dwarl stop runs/<run>`, and resume
with `dwarl train --resume runs/<run> --max-iterations 5000`. The iteration budget
is the total target, not additional iterations. Resume requires a completed checkpoint.
