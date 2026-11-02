# Test and understand results

**Goal:** evaluate a completed checkpoint and inspect how it navigates.

## Run a validation workload

```bash
pixi run test --suite validation
```

The selected robot's navigation run is preferred over its smoke run. DWARL prints
the resolved run, iteration and hash before executing. For explicit selection:

```bash
pixi run test --run runs/<run> --checkpoint best --suite validation
```

`best` requires completed periodic validation during training. An iteration number
pins a checkpoint; `latest` selects the latest completed checkpoint in the selected run.

## Read the result

```bash
pixi run dwarl report --open
```

Start with completed cases, success and collision rates. Inspect failed episodes,
their trajectory and command/measured-motion plots. Missing measurements are
shown as unavailable. A finished workload is not automatically a passed workload.

??? example "What an episode plot looks like"

    ![Example testbot smoke-run trajectory in a doorway case](assets/example-trajectory.png)

    This is an illustrative smoke run of the public testbot fixture, not a navigation-quality result.

| Exit code | Meaning |
| --- | --- |
| `0` | Workload passed |
| `1` | Completed workload failed navigation criteria |
| `2` | Input or infrastructure failure |
| `130` | Interrupted; available partial results retained |

## Expand evaluation

```bash
pixi run test --case doorway --show
pixi run test --sim both --suite validation
```

Gazebo requires a [checked adapter](gazebo.md). Default tests run at most 32 cases
per simulator; `--set test.max_cases=3` gives a shorter check.

Choose candidates on **validation**. Once the candidate is frozen, run the held-out
test suite with an explicit run/checkpoint:

```bash
pixi run test --run runs/<run> --checkpoint <iteration> --suite test
```

[Detailed metrics](results.md) · [Compare controllers](compare.md)

**Next:** [deploy the selected checkpoint](deploy.md).
