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
        for name, digest in expected.items():
            if name == "robot.xml" or name.startswith("assets/"):
                path = Path(temp) / "robot" / name
                if not path.is_file() or sha256(path) != digest:
                    raise ValueError(f"target model geometry changed since training: {name}")


def execute(cfg):
    if not cfg.robot:
        raise ValueError("select the target integration: dwarl deploy <robot>")
    checkpoint = resolve_checkpoint(resolve_run(cfg.deploy.run), cfg.deploy.checkpoint)
    profile = load_profile(cfg.robot)
    check_compatibility(checkpoint, profile)
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
    print(f"Installed {checkpoint.sha256} for {profile.id}: {target}")
    print(f"Selected policy: {selection}; recorded tests: {len(tests)}")
    launch = settings.get("launch")
    if launch:
        print(f"Launch: ros2 launch {launch['package']} {launch['file']}")
    else:
        print(f"Integration has no deployment launch adapter; runtime bundle: {target}")
    return target
