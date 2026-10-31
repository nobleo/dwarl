"""Generate discoverable command help and resolved defaults from the implementation."""

import argparse
import sys
import tempfile
from pathlib import Path

import mkdocs_gen_files
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dwarl.cli import parser  # noqa: E402
from dwarl.config import compose_config  # noqa: E402
from dwarl.robot.schema import ProfileCfg  # noqa: E402

root = parser()
with mkdocs_gen_files.open("reference.md", "w") as out:
    out.write("# Command reference\n\nGenerated from the CLI and resolved configuration on every site build.\n\n")
    out.write(f"```text\n{root.format_help()}```\n\n")
    for action in root._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, command in action.choices.items():
                out.write(f"## {name}\n\n```text\n{command.format_help()}```\n\n")
    out.write("## Default configuration\n\nOverride any setting with `--set path=value`. "
              "Manager parameters and rewards are included below.\n\n```yaml\n")
    with tempfile.TemporaryDirectory() as directory:
        out.write(OmegaConf.to_yaml(compose_config(cwd=directory), resolve=True))
    out.write("```\n")
    out.write("\n## Robot profile schema\n\nSee [the integration guide](integrate-robot.md) and "
              "[profile template](examples/dwarl_robot.yaml) for a starting point. "
              "Below, `???` marks required values; `null` means unset. "
              "Exactly one description source and one footprint representation are required.\n\n```yaml\n")
    out.write(OmegaConf.to_yaml(OmegaConf.structured(ProfileCfg)))
    out.write("```\n")
