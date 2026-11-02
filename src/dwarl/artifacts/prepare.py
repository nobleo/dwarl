"""The only checkpoint-to-runtime conversion path, shared by test and deploy."""

import json
import tempfile
from pathlib import Path

from dwarl.runs.store import identity, lock, sha256, write_json

EXPORTER_VERSION = 1


def validate_bundle(path):
    from dwarl_runtime import Policy

    data = json.loads((path / "provenance.json").read_text())
    actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
    if actual != set(data["files"]) | {"provenance.json"}:
        raise ValueError(f"bundle contains missing or unrecorded files: {path}")
    required = {"policy.onnx", "deploy.yaml", "dwarl_robot.yaml", "robot.xml", "manifest.json"}
    if not required <= actual:
        raise ValueError(f"incomplete runtime bundle: {path}")
    for relative, expected in data["files"].items():
        file = (path / relative).resolve()
        if not file.is_relative_to(path.resolve()) or sha256(file) != expected:
            raise ValueError(f"bundle file missing or changed: {relative}")
    Policy(path)
    return data


def prepare_policy(checkpoint, cache=None):
    if sha256(checkpoint.path) != checkpoint.sha256 or sha256(checkpoint.run / "contract.yaml") != checkpoint.contract:
        raise ValueError("selected checkpoint or saved training contract changed")
    key = identity({"checkpoint": checkpoint.sha256, "contract": checkpoint.contract,
                    "exporter": EXPORTER_VERSION})
    cache = Path(cache or Path.cwd() / ".dwarl/policies").resolve()
    output = cache / key
    with lock(cache / f".{key}.lock"):
        if not output.exists():
            from omegaconf import OmegaConf
            from rsl_rl.runners import OnPolicyRunner

            from dwarl.artifacts.exporter import export_runner
            from dwarl.testing.scenarios import SPLIT, build_scenarios
            from dwarl.training.contract import load_contract
            from dwarl.training.environment.nav_env import NavVecEnv

            cfg = load_contract(checkpoint.run)
            cfg.env.num_envs, cfg.sim.gui, cfg.device = 1, False, "cpu"
            cfg.curriculum.enabled = False
            with tempfile.TemporaryDirectory(prefix=".policy-", dir=cache) as temp:
                stage = Path(temp) / "bundle"
                dataset = Path(temp) / "reconstruction"
                build_scenarios(dataset, names=["open_space"], robot=cfg.robot.profile_file, previews=False)
                cfg.dataset.path, cfg.dataset.split = str(dataset), SPLIT
                env = NavVecEnv(cfg, split=SPLIT)
                try:
                    runner = OnPolicyRunner(env, OmegaConf.to_container(cfg.agent, resolve=True),
                                            log_dir=None, device="cpu")
                    runner.load(str(checkpoint.path), map_location="cpu")
                    export_runner(runner, cfg, env, stage, checkpoint.iteration, str(checkpoint.path))
                finally:
                    env.close()
                write_json(stage / "provenance.json", {
                    "schema": "dwarl_policy/1", "identity": key, "checkpoint": checkpoint.sha256,
                    "iteration": checkpoint.iteration, "contract": checkpoint.contract,
                    "exporter": EXPORTER_VERSION,
                    "files": {str(p.relative_to(stage)): sha256(p) for p in stage.rglob("*") if p.is_file()},
                })
                validate_bundle(stage)
                stage.rename(output)
        data = validate_bundle(output)
        if data["identity"] != key:
            raise ValueError("policy cache identity mismatch")
    return output
