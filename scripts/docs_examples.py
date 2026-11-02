"""Regenerate robot-neutral documentation illustrations through pixi.

MUJOCO_GL=egl pixi run python scripts/docs_examples.py [--test-directory <testbot-smoke-test>]
"""

import argparse
import shutil
from pathlib import Path

import mujoco
from PIL import Image

from dwarl.robot.build import build_xml
from dwarl.robot.profile import load_profile
from dwarl.robot.view import inspection_model

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-directory", type=Path)
    args = parser.parse_args()
    output = ROOT / "docs/assets"
    output.mkdir(exist_ok=True)
    profile = load_profile(ROOT / "tests/robots/testbot")
    model, data = inspection_model(build_xml(profile))
    model.vis.global_.offwidth, model.vis.global_.offheight = 700, 400
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, .15]
    camera.distance, camera.azimuth, camera.elevation = 1.35, 130, -25
    with mujoco.Renderer(model, height=400, width=700) as renderer:
        renderer.update_scene(data, camera)
        Image.fromarray(renderer.render()).save(output / "testbot-model.png", optimize=True)
    if args.test_directory:
        trajectory = next(args.test_directory.glob("episodes/*/mujoco/trajectory.png"))
        shutil.copyfile(trajectory, output / "example-trajectory.png")


if __name__ == "__main__":
    main()
