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

## Watch in the dashboard

```bash
pixi run web
```

The **Workflow** menu runs the same datagen, train, test, deploy, environment check,
status and next-step commands as the CLI. Select a workflow and its options,
then follow the logs, metrics and reports in the browser.
Switch on **Viewer** to watch the run in MuJoCo Studio in your browser.
For MuJoCo tests, **Real time** slows the run to wall-clock speed while you watch;
scores are unchanged and full speed resumes when the viewer closes.
**Documentation** opens this site from the dashboard in a new tab.

### Personalize and share the dashboard

In **Metrics**, open **Metric library** to inspect any scalar in the selected
run. Choose **Pin to wall** to keep it. Drag the dotted handles (or focus a
handle and press Up/Down) to reorder your wall. Pins survive switching runs.

Under **Workspace → Settings overrides**, search **+ Add setting** to pin a
training, recipe or test field. Pinning alone inherits the workspace/recipe
value. Editing creates an explicit CLI override; **Reset to inherited** removes
the value but keeps the control. Drag handles to order the controls. Run and
checkpoint selection stay in the command centre; preview checks use the CLI.

The server autosaves these preferences, including explicit overrides, to
`web/dashboard-setup.json` relative to the workspace where you started the
dashboard. All browsers connected to that workspace share this file. Under
**Dashboard setup**, **Save setup** writes it now, **Load setup** rereads it,
and **Export JSON** / **Import setup** transfer a copy. Loading a setup does
not start a job. To share it with a colleague, copy the file into their `web/`
folder or export and import it. Git ignores it by default; use
`git add -f web/dashboard-setup.json` to track an intentional team setup.

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
