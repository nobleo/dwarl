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
and checks the requested environment. Python, MuJoCo 4, ROS 2 Jazzy and Gazebo Harmonic
come from pixi. Both simulation environments (`default` and `train`) use the locked
MuJoCo 4 release. No manual sourcing is needed. Supported environment: **Linux x86-64**.

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

### Local web workspace

```bash
pixi run web-build          # build Vue + Vite + Tailwind + DaisyUI once
pixi run dwarl web          # serve locally and open the browser
```

The workspace runs the same train/test/deploy commands, with live output, run
progress, case outcomes, and native charts read directly from TensorBoard events:
reward, success/collision rates, path completion, losses, validation and throughput.
The browser embeds **MuJoCo 4 Studio**, including its interactive camera, visualization
settings and inspector, with DWARL's path markers and an environment selector.
Graphics run in the browser (WASM/Filament); the simulation host needs no GPU or display.
Live state is sampled at up to 20 Hz; Studio renders independently in the browser.
Viewing is **off by default** and its server/state streaming stops within approximately
three seconds when hidden, scrolled away or disconnected. Dashboard-launched MuJoCo jobs support viewing;
existing terminal runs still expose recorded progress, metrics and results.
Keep viewing off for compute measurements. Studio observes the running policy:
simulation edits, reset/pause and model loading do not alter the run. Use DWARL's
workflow controls to manage execution. Overview shows the selected shard in sharded training.

Training and evaluation have independent command slots: manually test or benchmark
completed checkpoints while training continues. The checkpoint dropdown lists
iterations and the best-validation marker; latest/best are pinned to an exact
iteration when the command starts. Choose **Test a checkpoint → Benchmark** for
the benchmark suite. Select an evaluation to review outcomes and its full report
with plots. Run final compute measurements separately from training.

The workspace also includes robot readiness checks, a preflight/settings preview,
same-workload checkpoint comparisons, recorded case replay and exact-case reruns,
and persistent command history. Reports and **Documentation** use native dashboard
views with search and generated CLI/configuration references; workflow help buttons
jump to the relevant guide. All frontend sources live in `web/`: Vue in `web/src/`,
documentation in `web/docs/`, and shared styling/logos in `web/shared/`.

One training and one evaluation command run at a time. Their **Stop** buttons
gracefully interrupt only that command;
closing a browser tab leaves it running. Ctrl+C in `dwarl web` stops the server,
its commands. Logs remain under `.dwarl/web/`.
The server binds only to `127.0.0.1`; use its printed URL, which contains the
session token. `--no-open` suppresses browser launch; `--port 0` selects a free port.
After changing backend code, start a new `dwarl web --port 0` instance. It reconnects
running dashboard jobs for inspection/control, while their original server keeps
ownership; closing the new server does not stop those reconnected jobs.

Frontend development: `pixi run npm --prefix web run dev` with `dwarl web` on port
8765, then open the Vite URL with the same `#token=…` fragment. Rebuild with
`pixi run web-build` before serving or packaging the production dashboard.

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
