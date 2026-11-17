"""Build the shared documentation into the distributable dashboard assets."""

from pathlib import Path

from mkdocs.commands.build import build
from mkdocs.config import load_config

ROOT = Path(__file__).resolve().parents[2]
build(load_config(config_file=str(ROOT / "web/mkdocs.yml"),
                  site_dir=str(ROOT / "src/dwarl/web/static/docs"), strict=True))
