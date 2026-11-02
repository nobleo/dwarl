from concurrent.futures import ThreadPoolExecutor

import pytest
from omegaconf import OmegaConf

from dwarl.artifacts.prepare import prepare_policy, validate_bundle
from dwarl.robot.config import materialize
from dwarl.runs.store import complete_checkpoint, resolve_checkpoint, write_json


@pytest.fixture
def checkpoint(train_cfg, tmp_path):
    from rsl_rl.runners import OnPolicyRunner

    from dwarl.training.environment.nav_env import NavVecEnv

    run = tmp_path / "run"
    run.mkdir()
    materialize(train_cfg, run / "robot")
    OmegaConf.save(train_cfg, run / "contract.yaml", resolve=True)
    env = NavVecEnv(train_cfg)
    try:
        runner = OnPolicyRunner(env, OmegaConf.to_container(train_cfg.agent, resolve=True), log_dir=None)
        path = run / "checkpoint.pt"
        runner.save(str(path))
    finally:
        env.close()
    write_json(run / "checkpoint_manifest.json", {"schema": "dwarl_checkpoints/1", "completed": [], "best": None})
    complete_checkpoint(run, path, 0)
    return resolve_checkpoint(run)


def test_two_consumers_reuse_one_completed_bundle(checkpoint, tmp_path, monkeypatch):
    from dwarl.artifacts import exporter

    original = exporter.export_runner
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(exporter, "export_runner", counted)
    cache = tmp_path / "cache"
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: prepare_policy(checkpoint, cache), range(2)))
    assert results[0] == results[1]
    assert len(calls) == 1
    assert validate_bundle(results[0])["checkpoint"] == checkpoint.sha256
    assert not (results[0] / "policy.pt").exists()


def test_failed_conversion_has_no_published_bundle(checkpoint, tmp_path, monkeypatch):
    from dwarl.artifacts import exporter

    def fail(runner, cfg, env, out, *args):
        out.mkdir()
        (out / "policy.onnx").write_bytes(b"partial")
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(exporter, "export_runner", fail)
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="conversion failed"):
        prepare_policy(checkpoint, cache)
    assert not [p for p in cache.iterdir() if p.is_dir()]


def test_declared_contract_change_is_rejected(checkpoint):
    with (checkpoint.run / "contract.yaml").open("a") as stream:
        stream.write("\n# modified after training\n")
    with pytest.raises(ValueError, match="contract has changed"):
        resolve_checkpoint(checkpoint.run)
