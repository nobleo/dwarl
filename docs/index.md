<div class="docs-hero" markdown>

![DWARL — dynamic window approach and reinforcement learning](assets/dwarl-logo.png#only-light){ .hero-logo }
![DWARL — dynamic window approach and reinforcement learning](assets/dwarl-logo-dark.png#only-dark){ .hero-logo }

<span class="eyebrow">NOBLEO TECHNOLOGY · DWARL</span>

# Reinforcement learning within the dynamic window

Train a local planner in MuJoCo, measure how it navigates, and bring the
same policy to ROS 2 Nav2.

[Try the example robot →](quickstart.md){ .md-button .md-button--primary }
[Use your robot →](integrate-robot.md){ .md-button }

<span class="hero-note">Linux x86-64 · ROS 2 Jazzy · Experimental · No pretrained policy included</span>

</div>

## Start with one command

Install [pixi](https://pixi.sh), clone the repository and let setup guide you:

```bash
git clone https://github.com/nobleo/dwarl.git
cd dwarl
pixi run setup
```

Pixi manages Python, ROS and Gazebo. Setup remembers your robot. Every command
shows its result and a next step.

Prefer to use your coding agent? [Give it the included setup skill](agent-setup.md)
with a ready-to-paste prompt.

<div class="grid cards" markdown>

- **01 · Train**

    Build robot-sized datasets and train a policy. Start with a tiny pipeline check.

    [Train a policy →](train.md)

- **02 · Test**

    Evaluate exact checkpoints and inspect success, collision and per-case behavior.

    [Understand results →](test.md)

- **03 · Deploy**

    Install a complete ONNX bundle and connect the controller to Nav2.

    [Deploy to Nav2 →](deploy.md)

</div>

## Pick up where you left off

```bash
pixi run dwarl status
pixi run dwarl next
```

DWARL replaces the **local controller**. Your robot supplies its description,
drivers, lidar, odometry and localization. Start in simulation; choose policies
using validation evidence rather than training reward alone.

[CLI reference](reference.md) · [Troubleshooting](troubleshooting.md) · [GitHub](https://github.com/nobleo/dwarl)
