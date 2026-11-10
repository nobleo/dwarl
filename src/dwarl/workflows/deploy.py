import json
import shutil
import tempfile
from pathlib import Path

from dwarl.artifacts.prepare import prepare_policy, validate_bundle
from dwarl.robot.profile import load_profile
from dwarl.runs.store import identity, lock, resolve_checkpoint, resolve_run, write_json


def check_compatibility(checkpoint, profile):
    source = load_profile(checkpoint.run / "robot/source_profile.yaml")
    # Installation metadata is not a physical robot fact.
    def physical(p):
        return {k: v for k, v in p.raw.items() if k not in ("deployment", "datagen")}
    if identity(physical(profile)) != identity(physical(source)):
        raise ValueError(f"target robot is incompatible with checkpoint {checkpoint.sha256}")
    from dwarl.robot.config import materialize
    from dwarl.runs.store import sha256
    from dwarl.training.contract import load_contract

    # A profile can stay unchanged while a referenced URDF/mesh changes.
    with tempfile.TemporaryDirectory(prefix="dwarl-target-") as temp:
        contract = load_contract(checkpoint.run)
        contract.robot.profile = str(profile.path)
        materialize(contract, Path(temp) / "robot")
        frozen = checkpoint.run / "robot"
        expected = json.loads((frozen / "manifest.json").read_text())["files"]
        changed = [name for name, digest in expected.items()
                   if (name == "robot.xml" or name.startswith("assets/"))
                   and not ((Path(temp) / "robot" / name).is_file() and sha256(Path(temp) / "robot" / name) == digest)]
        if changed:  # the text differs; a newer MuJoCo may write the same model differently
            difference = model_difference(frozen / "robot.xml", Path(temp) / "robot" / "robot.xml")
            if difference:
                raise ValueError(f"target model geometry changed since training: {difference}")


# Compiled-model fields that label a model without changing its physics. MuJoCo 4 writes input="vel" for
# velocity actuators (actuator_ctrlspec); models written by MuJoCo 3 leave it out, with the same gain and bias.
_LABELS = {"actuator_ctrlspec"}


def model_difference(trained, target, tolerance=1e-9):
    """First compiled-model field that differs between two MJCF files, or None if they are the same model.

    Both files are compiled with the installed MuJoCo, so differences in how a MuJoCo version writes the XML
    (number formatting, defaults written out) do not count; geometry, inertia, joints and actuators do.
    """
    import mujoco
    import numpy as np

    if not target.is_file():
        return target.name
    a, b = (mujoco.MjModel.from_xml_path(str(path)) for path in (trained, target))
    fields = [(name, getattr(a, name), getattr(b, name)) for name in sorted(dir(a))
              if not name.startswith("_") and name not in _LABELS]
    fields += [(f"opt.{name}", getattr(a.opt, name), getattr(b.opt, name)) for name in sorted(dir(a.opt))
               if not name.startswith("_")]  # timestep, gravity, solver
    for name, x, y in fields:
        if isinstance(x, np.ndarray):
            numeric = x.dtype.kind in "biuf"
            same = x.shape == y.shape and (np.allclose(x, y, rtol=tolerance, atol=tolerance) if numeric
                                           else np.array_equal(x, y))
        elif isinstance(x, (bool, int, float)):
            same = np.isclose(x, y, rtol=tolerance, atol=tolerance)
        else:
            continue
        if not same:
            return name
    return None


def execute(cfg):
    from dwarl.cli import ui

    if not cfg.robot:
        raise ValueError("select the target integration: dwarl deploy <robot>")
    checkpoint = resolve_checkpoint(resolve_run(cfg.deploy.run), cfg.deploy.checkpoint)
    profile = load_profile(cfg.robot)
    check_compatibility(checkpoint, profile)
    ui.finding("pass", f"Target compatible with checkpoint {checkpoint.iteration} · {checkpoint.sha256[:12]}")
    if profile.raw["deployment"].get("launch"):
        from dwarl.workflows.prepare import ros

        ros(profile.path)
    bundle = prepare_policy(checkpoint)
    metadata = validate_bundle(bundle)
    settings = profile.raw["deployment"]
    directory = (profile.dir / settings["policy_directory"]).resolve()
    selection = (profile.dir / settings["selected_policy"]).resolve()
    if not directory.is_relative_to(profile.dir) or not selection.is_relative_to(profile.dir):
        raise ValueError("deployment paths must remain inside the integration directory")
    target = directory / metadata["identity"]
    with lock(directory / ".install.lock"):
        if target.exists():
            if validate_bundle(target) != metadata:
                raise FileExistsError(f"different content already installed: {target}")
        else:
            with tempfile.TemporaryDirectory(prefix=".install-", dir=directory) as tmp:
                stage = Path(tmp) / "bundle"
                shutil.copytree(bundle, stage)
                validate_bundle(stage)
                stage.rename(target)
        tests = []
        for file in (checkpoint.run / "tests").glob("*/summary.json"):
            result = json.loads(file.read_text())
            if result["checkpoint"] == checkpoint.sha256:
                tests.append({"summary": str(file), "status": result["status"], "passed": result["passed"]})
        # JSON is a YAML subset, allowing the same atomic writer and easy ROS-side parsing.
        write_json(selection, {"schema": "dwarl_selected_policy/1", "bundle_id": metadata["identity"],
                   "path": str(target.relative_to(profile.dir)), "checkpoint": checkpoint.sha256, "tests": tests})
    ui.finding("pass", f"Installed policy for {profile.id}: {target}")
    ui.info(f"Selected policy: {selection} · recorded evaluations: {len(tests)}")
    launch = settings.get("launch")
    if launch:
        ui.info("After starting robot interfaces and localization:")
        ui.info(ui.command("ros2", "launch", launch["package"], launch["file"], f"integration:={profile.dir}"))
    else:
        ui.info(ui.command("ros2", "launch", "dwarl_bringup", "navigation.launch.py", f"model_dir:={target}"))
    ui.info("Transfer the whole bundle to another computer; use the pixi runtime environment on supported targets.")
    return target
