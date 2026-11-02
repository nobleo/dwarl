---
name: dwarl-robot-setup
description: Integrate a user's differential-drive robot with DWARL using pixi, from an existing profile or URDF/xacro through setup, simulator checks and a bounded training/ONNX smoke test. Use when asked to set up a robot, onboard an existing robot workspace, create a DWARL integration, or diagnose its sensor/command contract. Ask for missing physical measurements rather than borrowing example values.
---

# Robot setup for DWARL

This is a portable workflow for any coding agent with file and terminal access.
Run commands from the DWARL checkout. Paths below are relative to that checkout,
even if this skill has been copied into another agent's skill directory.

## Goal and boundaries

Finish with a selected, checked integration and a small reproducible pipeline test.
For a setup-only request, stop there and give the next command. Long navigation
training, controller tuning, installing a replacement policy and driving hardware
are separate tasks. A smoke policy is not a navigation-quality policy.

Read `AGENTS.md`, `docs/integrate-robot.md`, and any instructions in the robot
repository. Use `pixi run ...` throughout; never ask users to source ROS overlays.
Check `--help` against the installed checkout if an option differs from this guide.

## 1. Locate the robot and inspect existing work

- Establish the DWARL checkout and robot workspace paths. Ask which workspace to
  use if multiple candidates match; do not select the first sibling repository.
- Inspect git status in both repositories. Preserve user edits, historical runs,
  installed policies and launch customizations. Keep robot-specific files and
  results in the robot repository or ignored local artifacts, not public DWARL files.
- Find `dwarl_robot.yaml`, package manifests, URDF/xacro, meshes and launch files.
  Prefer an existing integration over regenerating it.
- Confirm pixi is available and the supported Linux x86-64 environment. If blocked,
  report the precise dependency or environment issue rather than claiming setup passed.

## 2. Establish the robot contract

Read the profile and source description/configuration. Establish:

| Input | What to verify |
| --- | --- |
| Description and drive | URDF/xacro arguments, base link, left/right wheel joints, radius and separation |
| Footprint | Physical polygon or radius, with its offset relative to the base origin |
| Limits | Forward/reverse speed, turn rate, linear/angular acceleration, control period |
| Lidar | Link, full mounting transform, filtered field of view, range, ray count and history |
| ROS | Base/odom/map frames, scan/odom/command topics, Twist versus TwistStamped |
| Simulation | Support/caster contact model, actuator assumptions, robot-only launch adapter |

Use values found in the robot's files and identify their sources. Ask the user for
missing physical measurements or conflicting values in one concise batch. A URDF
does not establish safe command limits. Do not fill gaps with another robot's
geometry, an inflated costmap radius, or the template's example numbers.
Upside-down/offset lidars affect ray direction and clearance. A rear safety polygon
does not provide rear sensing when the lidar only sees the front.

## 3. Create only what is missing, then select the integration

For an existing integration, run the normal non-interactive setup:

```bash
pixi run setup --robot-ws <robot-workspace> --packages <integration-package> --yes
```

For a new integration, prepare an answers YAML in the robot repository using
verified inputs and the schema in `src/dwarl/robot/schema.py`. Then use the generator:

```bash
pixi run dwarl robot init <robot-workspace>/src/<integration-package> \
  --answers <answers.yaml> --maintainer '<maintainer>' --email <email> --yes
pixi run setup --robot-ws <robot-workspace> --packages <integration-package> --yes
```

Get package metadata from the repository or user. Use `--profile-only` for a
standalone profile without ROS scaffolding; `robot init` selects it locally.
For this standalone branch, skip the package-based `setup` command above: it
discovers integrations inside ROS packages under the workspace's `src/` directory.
Relative asset paths in the answers YAML resolve beside that file. Use an existing
workspace's layout rather than moving packages to match the example above.

Do not launch the interactive wizard in a terminal that cannot answer prompts.
If the user wants to run it themselves, give `pixi run setup`. Interrupted wizard
drafts live in `.dwarl/drafts/`; rerun with the same destination to continue.
For updates, inspect the proposed diff before using `--update`; preserve edited
launch files. Add Gazebo configuration before training so the frozen contract matches.

## 4. Check the model and requested simulator

```bash
pixi run dwarl robot check <integration> --json
pixi run train <integration> --preset smoke --print-config
pixi run train <integration> --preset smoke --dry-run
```

Inspect geometry on a floor with `pixi run dwarl robot view <integration>` when a
GUI is available, or ask the user to do that step. Report visual inspection as
pending if it has not happened. Do not infer correct support contact from compilation.

If Gazebo is requested/configured, read `docs/gazebo.md`. Wrap an existing robot-only
launch with `robot init --gazebo` and explicit `--sim-package`, `--sim-launch` and
`--arg-map` options when needed; inspect `robot init --help` for the complete form.
Then run:

```bash
pixi run dwarl doctor --robot <integration> --sim gazebo --live --json
```

Verify clock, filtered scan, odometry, TF, command type, entity `robot` in world
`default`, nonzero spawn pose and zero-origin odometry. The live check sends no
motion commands and needs no policy. A wrapper file alone is not evidence it works.
For MuJoCo-only setup, skip Gazebo and run `pixi run dwarl doctor --json` instead.

Read dependency warnings. Distinguish packages needed by the selected launch from
optional broader bringup dependencies. When required, use
`pixi add --feature ros ros-jazzy-<pkg>` and rerun setup; retain the lockfile change.
Run Gazebo sequentially, preserve user-started simulators, and inspect launch logs
on failures. Respect the shared build/simulator locks.

## 5. Exercise a bounded pipeline

```bash
pixi run train <integration> --preset smoke --json
pixi run test --run <returned-run> --checkpoint <completed-iteration> \
  --sim mujoco --suite validation --set test.max_cases=3 --json
pixi run dwarl report <returned-test-directory> --json
```

Take the run path and completed iteration from actual output, not a guessed name
or global `latest`. Record the checkpoint hash. Verify robot-sized dataset metadata
and keep training, validation and held-out cases distinct. This small check exports
ONNX and exercises inference; do not install the smoke bundle over an existing policy.

Read `summary.json`, `episodes.json` and available per-case traces. Exit `1` means
completed navigation failures; `2` means input/infrastructure failure; `130` means
interrupted. Report collisions honestly even when expected from a two-iteration policy.
Do not loosen scoring or start an unbounded tuning loop to make setup look successful.

If policy-driven Gazebo testing is requested, start with one pinned validation case
before broader tests. Read `compute.json` real-time factor and `launch.log` lifecycle
failures before interpreting outcomes. MPPI/DWB comparisons require Gazebo, not
MuJoCo. Desktop inference timing is not physical-robot performance.

## 6. Hand off evidence and one next step

Use `pixi run dwarl status --json` and `pixi run dwarl next --json` to inspect saved
state. JSON results are on stdout; human diagnostics are on stderr. Do not parse
terminal tables or treat a process exit alone as evidence of successful navigation.

Finish with:

- Selected integration and files/settings changed.
- Checks passed, failed or skipped, including pending visual inspection.
- Smoke run, checkpoint iteration/hash, test outcomes and report path.
- Outstanding inputs/dependencies and the exact next command.

For a completed setup, the usual next command is `pixi run train --preset navigation`.
Use `.agents/skills/dwarl-performance-loop/SKILL.md` for later evidence-led tuning.
For deployment requests, follow `docs/deploy.md` and check compatibility with the
target profile; never blindly reuse another robot's policy or dataset.
