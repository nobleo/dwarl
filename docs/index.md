<div class="docs-hero" markdown>

<span class="eyebrow">NOBLEO TECHNOLOGY · DWARL</span>

# From training to navigation.

Build a neural local planner for your robot. Train in MuJoCo, compare with
Gazebo, and deploy the same policy to Nav2.

[Install DWARL →](#install){ .md-button .md-button--primary }
[Explore the commands](reference.md){ .md-button }

<span class="hero-note">Experimental toolkit · ROS 2 Jazzy · No pretrained policy included</span>

</div>

<div class="workflow-grid" markdown>

<a class="workflow-card" href="configuration/">
<span class="step-label">01 / TRAIN</span>
<strong>Start with your robot</strong>
<span>Inspect the model, generate robot-sized maps, and run your first training session.</span>
<code>dwarl train</code>
</a>

<a class="workflow-card" href="results/">
<span class="step-label">02 / TEST</span>
<strong>Measure what works</strong>
<span>Evaluate exact checkpoints and compare trajectories across both simulators.</span>
<code>dwarl test</code>
</a>

<a class="workflow-card" href="ros/">
<span class="step-label">03 / DEPLOY</span>
<strong>Bring it to your robot</strong>
<span>Install a tested policy into your robot integration and run it with Nav2.</span>
<code>dwarl deploy</code>
</a>

</div>

## What can I do?

| Goal | Command / guide |
| --- | --- |
| Inspect robot geometry on a floor | `dwarl robot view <robot-repository>` |
| Check model parameters | `dwarl robot check <robot-repository>` |
| Check simulator installation | `dwarl doctor --sim mujoco` or `--sim gazebo` |
| Generate robot-sized data and train | `dwarl train <robot-repository>` |
| Watch training | Add `--show`; [monitor TensorBoard](configuration.md#monitor-training) |
| Run in the background | Add `--detached`; stop with `dwarl stop <run>` |
| List and resume runs | `dwarl runs list`; `dwarl train --resume <run>` |
| Evaluate canonical cases | `dwarl test --run <run> --suite benchmark` |
| Evaluate validation / held-out data | `--suite validation` / `--suite test` |
| Compare MuJoCo and Gazebo | `dwarl test --run <run> --sim both` |
| Watch a case | `dwarl test --run <run> --case doorway --show` |
| Install a checkpoint into a robot | `dwarl deploy <robot-repository> --run <run>` |
| See all flags and settings | [Generated command reference](reference.md) |
| Understand scores and output files | [Results](results.md) |

Use `--help` on any command. Train, test and deploy also offer `--print-config`
and `--dry-run`. [Configuration](configuration.md) explains overrides and selection.

## Install

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/nobleo/dwarl.git
cd dwarl
uv sync --group dev
source .venv/bin/activate
dwarl --help
```

Repository access is required while the project is private. For Gazebo/Nav2,
follow [ROS setup](ros.md) and build your robot workspace first.
