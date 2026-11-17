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

## Interactive browser viewer

```bash
pixi run web-build
pixi run dwarl web
```

Start or resume training, or start a MuJoCo checkpoint test, from the workspace.
Enable **MuJoCo Studio** to see the actual running simulation. Studio's WASM/Filament
client renders in your browser, with its native camera navigation, visualization
settings, inspector and fullscreen support. The Python host needs neither a GPU nor
a display server. The dashboard proxies Studio's HTTP and WebSocket connections
through its authenticated port; no second port needs forwarding for remote use.
For an SSH-hosted workspace, forward the dashboard port to the same local port
and open its printed `127.0.0.1` URL locally.

DWARL sends immutable snapshots at up to 20 Hz and retains ownership of policy
execution and physics. Studio does not advance a second simulation. Its simulation
edits, pause/reset and model-loading actions do not change training or test results;
use the workspace's workflow controls. Path, goal and lookahead markers accompany
the live robot. The environment selector can access every training shard; overview
shows the selected shard.

Disabling the viewer removes its browser canvas and closes Studio's server.
Scrolling away, switching tabs or disconnecting expires the viewing lease after
approximately three seconds, even during a long policy update. Physics continues.
Enabling viewing again starts a fresh Studio session. This removes state-copying,
streaming and UI work while viewing is off; throughput gains depend on the workload.
The desktop viewer is independent and must be closed separately.

After updating the backend, start a new `dwarl web --port 0` instance. Jobs launched
with the earlier image-stream viewer need to be restarted/resumed to load the
Studio hook. Keep an older owning server running until its jobs finish.

## Background execution

```bash
pixi run train --preset navigation --detached
```

The command prints a process log path. Once model/data preparation completes, the
run is available through `runs list` and `runs watch`. `runs view` toggles a live
simulator viewer on the running training without stopping it.

## Stop and continue

```bash
pixi run dwarl stop runs/<run>
pixi run train --resume runs/<run> --max-iterations 5000
```

The budget is total iterations, not additional iterations. Resume restores the saved
recipe and completed checkpoint; simulator episodes restart. Ctrl+C also retains
completed checkpoints and available partial test evidence.

[Early stopping and detailed metrics](configuration.md#optional-intermediate-testing-and-early-stopping)
