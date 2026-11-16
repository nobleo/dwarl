"""Composing a configuration must not need numpy: the documentation build renders the defaults in a
slim environment without the simulation stack."""

import subprocess
import sys

BLOCK_AND_COMPOSE = """
import sys
sys.modules["numpy"] = None          # any `import numpy` now raises ImportError
from dwarl.config import compose_config
from dwarl.config.presets import PRESETS
compose_config()
for preset in PRESETS:
    compose_config(preset=preset)
print("ok")
"""


def test_compose_config_without_numpy():
    out = subprocess.run([sys.executable, "-c", BLOCK_AND_COMPOSE], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")
