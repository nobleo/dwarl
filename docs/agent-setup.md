# Set up with your coding agent

Your coding agent can inspect your robot repository, run DWARL setup and verify
the first training/inference cycle. DWARL includes a **portable setup skill**:
plain Markdown instructions that work with any agent able to read files and run
terminal commands.

## 1. Open the checkout

Install [pixi](https://pixi.sh), clone DWARL and open your coding agent in that folder:

```bash
git clone https://github.com/nobleo/dwarl.git
cd dwarl
```

Give the agent access to your robot repository too. DWARL's environment currently
supports Linux x86-64. Bring your URDF/xacro, footprint, velocity/acceleration limits
and lidar settings; the agent can find existing values but will ask for missing measurements.

## 2. Paste this prompt

Replace the workspace path and choose whether you need Gazebo:

```text
Help me integrate my robot with DWARL.

Read AGENTS.md and .agents/skills/dwarl-robot-setup/SKILL.md in this
DWARL checkout and follow the setup skill.

My robot workspace is: /absolute/path/to/my_robot
I want: MuJoCo training and a live Gazebo interface check.

Inspect my existing profile, URDF and launch files first. Ask me for
missing measurements rather than guessing. Preserve existing edits
and policies. Run setup, model checks, the live simulator check if
applicable, and a bounded smoke training/inference test through pixi.

Finish with what changed, checks passed/failed/skipped, the report
path and the next command. Stop after the setup smoke test.
```

For a training-only integration, change the preference to **MuJoCo only**.
An existing `dwarl_robot.yaml` is reused; a new integration is generated in your
robot repository with `dwarl robot init`.

## 3. Follow the agent's questions

The agent checks the model and interfaces, saves the robot selection, and runs a
two-iteration smoke recipe. Gazebo users also get a policy-independent check of
clock, scan, odometry, TF, spawn pose and command subscription.
The smoke policy test runs in MuJoCo. To also test policy-driven navigation in
Gazebo, ask the agent for one pinned Gazebo validation case after the interface check.

Expect a short results summary and a report path. Smoke navigation failures are
possible and do not imply an installation failure. Missing dependencies and failed
interface checks should be reported separately. Inspect the model on a floor when
prompted; a compiled model alone does not verify caster/support geometry.

After setup:

```bash
pixi run dwarl status
pixi run dwarl next
```

## Skill discovery and reuse

The source lives at `.agents/skills/dwarl-robot-setup/SKILL.md`, and `AGENTS.md`
points agents to it. Some agents discover this directory automatically; the explicit
prompt above works even when yours does not. No special plugin or MCP server is needed.

If your agent supports installed skills, copy the `dwarl-robot-setup` directory
into its documented skill location. Keep it updated with your DWARL checkout and
run commands from that checkout. Otherwise, simply ask the agent to read the file.

[Read or download the setup instructions](agent-skills/dwarl-robot-setup.txt)
— generated from the same source used by agents in the repository.

**Continue:** [Train a policy](train.md) · [Manual setup](integrate-robot.md)
