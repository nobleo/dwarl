# Configuration

Precedence: built-in typed defaults → explicit `--preset` → `./dwarl.yaml` → command-line assignments.
`--config experiment.yaml` replaces the workspace file. YAML paths are relative to
that file; command-line paths are relative to the invoking directory. Duplicate CLI
assignments are errors. `--set` uses the actual schema key.

When a robot or run is not explicitly configured, the CLI uses setup's local
selection. Run pointers are scoped to that robot; navigation runs take precedence
over smoke runs. The selected inputs are printed before execution. Use `--run` to
override this choice. See [generated presets and defaults](reference/configuration.md).

```yaml
schema: dwarl/1
robot: /path/to/robot-repository
train:
  name: navigation
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

`dwarl train --resume runs/<name>` restores the saved recipe. Device, viewer,
training budget and whether early stopping is enabled may change. Simulator episodes restart; RNG/sampler bookkeeping is
restored, but continuation is not bitwise identical to uninterrupted physics.

Automatic data caches include robot clearance, generation parameters, seed and
generator version. An explicit `train.data.path` is checked for footprint compatibility;
`train.data.allow_custom=true` acknowledges an intentional custom dataset.

Periodic validation uses the validation split only. Best means maximum success rate
over a completed, identical validation workload; ties keep the earliest selection.
The held-out `test` suite is never used for checkpoint selection.

## Optional intermediate testing and early stopping

**Both are disabled by default.** A normal `dwarl train <robot>` runs to its
iteration budget without intermediate evaluation or automatic plateau stopping.
Enable only the checks you want:

```bash
dwarl train /path/to/robot-repository \
  --set train.validation_interval=200 \
  --set train.benchmark_interval=500
```

Each check runs the just-saved checkpoint headlessly in MuJoCo, in a separate
process. Training waits for it to finish. Validation uses the validation split;
benchmark uses the canonical scenarios. Neither uses the held-out test split.
Reports use the same format as `dwarl test` and live under `runs/<run>/tests/`.
Console output links to each report and to its `validation/` or `benchmark/` log.

Intervals are **absolute training iterations**, must be multiples of
`train.save_interval` (default 50), and use `0` to disable a check. Iteration 0
is skipped. `train.validation_cases` and `train.benchmark_cases` cap the workload
(both default to 32). A repeated final checkpoint save does not rerun the checks.

Only completed validation checks update `--checkpoint best`. Benchmark scores
never select checkpoints. Without periodic validation evidence, use `latest` or
an iteration number; manually running `dwarl test --suite validation` produces a
report but does not update training's best-checkpoint selection.

To also stop on a validation plateau:

```bash
dwarl train /path/to/robot-repository \
  --set train.validation_interval=200 \
  --set train.early_stop.enabled=true \
  --set train.early_stop.patience=5 \
  --set train.early_stop.min_delta=0.01
```

Early stopping requires periodic validation. The first completed check establishes
a baseline. Patience counts **completed validation checks**, not iterations.
An improvement of at least `min_delta` over the last significant best resets
patience (0.01 means one percentage point of success rate). Ties never reset it,
even with `min_delta=0`. Incomplete or infrastructure-failed checks do not consume
patience. Best-checkpoint selection still records any strictly better score,
independent of this stopping threshold. There is no reward-phase switching or
curriculum-level gate; the fixed validation workload drives the stop decision.

On stopping, the checkpoint already exists, `status.json` records
`state: early_stopped` and a human-readable reason, and TensorBoard records the
stop. Check history and patience are saved with checkpoint bookkeeping and
post-check `training_checks/<checkpoint-hash>.json` sidecars. Keep these sidecars
with the run when moving it: they preserve decisions made after the immutable
checkpoint was saved. Resume retains the schedule and patience. An already
early-stopped run stays stopped unless you explicitly disable early stopping:

```bash
dwarl train --resume runs/<run> --max-iterations 5000 \
  --set train.early_stop.enabled=false
```

Older runs retain their saved validation interval on resume; missing new options
default to disabled. A run already training in another process is not changed.

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

TensorBoard also shows:

- `Progress/*`: iteration, target budget and completed fraction.
- `Curriculum/*`: difficulty level, success-window progress, last window success,
  promotion/demotion thresholds and eligible pair count.
- `Validation/*` and `Benchmark/*`: whether checks are enabled, completion,
  success/collision rates and case counts. Validation also logs best score/iteration.
- `EarlyStop/*`: whether enabled, significant best, checks without improvement,
  remaining patience and the stop flag (when enabled).

Add `--show` to watch training or `--detached` to run it in the background.
List runs with `dwarl runs list`, stop with `dwarl stop runs/<run>`, and resume
with `dwarl train --resume runs/<run> --max-iterations 5000`. The iteration budget
is the total target, not additional iterations. Resume requires a completed checkpoint.
