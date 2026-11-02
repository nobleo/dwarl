"""Generate nested CLI help, schema tables, defaults and downloadable examples."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import mkdocs_gen_files
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dwarl.cli import parser  # noqa: E402
from dwarl.config import compose_config  # noqa: E402
from dwarl.config.presets import PRESETS  # noqa: E402
from dwarl.robot.schema import ProfileCfg  # noqa: E402


def commands(root, prefix="dwarl"):
    # Plain, deterministic help in documentation, independent of terminal capabilities.
    root.formatter_class = argparse.RawDescriptionHelpFormatter
    yield prefix, root
    for action in root._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                yield from commands(child, f"{prefix} {name}")


with mkdocs_gen_files.open("reference.md", "w") as out:
    out.write("# CLI reference\n\nGenerated from the command parser. Start with `pixi run setup`.\n\n"
              "Human diagnostics use stderr; `--json` produces structured stdout. `--plain` disables styling.\n\n"
              "Shell completion: inside `pixi shell`, run `eval \"$(dwarl completion bash)\"` "
              "(also supports zsh and fish).\n\n")
    for name, command in commands(parser()):
        if name.count(" ") == 1:
            out.write(f'<a id="{name.split()[1]}"></a>\n\n')
        out.write(f"## {name}\n\n```text\n{command.format_help()}```\n\n")
    out.write("## Default configuration\n\nSee [all defaults and presets](reference/configuration.md).\n\n"
              "## Robot profile schema\n\nSee the [robot profile reference](reference/robot-profile.md).\n")

with mkdocs_gen_files.open("reference/configuration.md", "w") as out:
    out.write("# All configuration defaults\n\nGenerated from the typed configuration. "
              "See [configuration precedence](../configuration.md).\n\n")
    out.write("## Explicit presets\n\nPresets apply before project YAML and CLI overrides.\n\n")
    for name, values in PRESETS.items():
        out.write(f"### {name}\n\n```yaml\n{OmegaConf.to_yaml(OmegaConf.create({'train': values}))}```\n\n")
    with tempfile.TemporaryDirectory() as directory:
        out.write("## Defaults\n\n```yaml\n" + OmegaConf.to_yaml(compose_config(cwd=directory), resolve=True) + "```\n")

with mkdocs_gen_files.open("reference/robot-profile.md", "w") as out:
    out.write("# Robot profile\n\nCreate one with `pixi run dwarl robot init`. "
              "The profile lives in your robot repository.\n\n"
              "[Download an example](../examples/dwarl_robot.yaml) · "
              "[Manual conventions](../manual-integration.md#2-add-a-robot-integration)\n\n"
              "`???` marks required measurements. A null default means unset; conditional requirements "
              "(such as wheel joints for URDF) are checked by the loader.\n\n"
              "| Field | Default |\n| --- | --- |\n")
    schema = OmegaConf.to_container(OmegaConf.structured(ProfileCfg))

    def fields(data, prefix=""):
        for key, value in data.items():
            name = prefix + key
            if isinstance(value, dict):
                yield from fields(value, name + ".")
            else:
                yield name, json.dumps(value)

    for key, value in fields(schema):
        out.write(f"| `{key}` | `{value}` |\n")
    out.write("\n## YAML schema\n\n```yaml\n" + OmegaConf.to_yaml(OmegaConf.structured(ProfileCfg)) + "```\n")

for name in ("dwarl_robot.yaml", "navigation.launch.py"):
    with mkdocs_gen_files.open(f"examples/{name}", "w") as out:
        out.write((Path(__file__).resolve().parents[1] / "src/dwarl/templates" / name).read_text())

with mkdocs_gen_files.open("agent-skills/dwarl-robot-setup.txt", "w") as out:
    out.write((Path(__file__).resolve().parents[1] / ".agents/skills/dwarl-robot-setup/SKILL.md").read_text())
