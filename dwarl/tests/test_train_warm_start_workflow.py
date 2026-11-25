"""Warm-start selection uses published evidence before creating a destination."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf

from dwarl.config import compose_config
from dwarl.runs.store import complete_checkpoint, resolve_run, write_json
from dwarl.workflows.train import execute


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    origin = tmp_path / "runs" / "source"
    origin.mkdir(parents=True)
    (origin / "config.yaml").write_text("{}")
    (origin / "contract.yaml").write_text("{}")
    write_json(origin / "provenance.json", {"schema": "dwarl_run/1", "created_ns": 1})
    write_json(origin / "checkpoint_manifest.json",
               {"schema": "dwarl_checkpoints/1", "completed": [], "best": None})
    for iteration in (10, 20):
        path = origin / f"model_{iteration}.pt"
        path.write_text(f"checkpoint {iteration}")
        complete_checkpoint(origin, path, iteration)

    cfg = compose_config(cwd=tmp_path, assignments=[
        "train.init_from=latest", "train.deployment_ready=false", "train.device=cpu"])
    cfg.robot = "testbot"
    contract = OmegaConf.create({"device": "cpu", "sim": {"gui": False}, "seed": 42,
                                "dataset": {"path": ""}, "agent": {}})
    profile = SimpleNamespace(id="testbot", path=tmp_path / "testbot", sha256="profile")
    monkeypatch.setattr("dwarl.robot.profile.load_profile", lambda _: profile)
    monkeypatch.setattr("dwarl.training.contract.build_contract", lambda _: contract)
    monkeypatch.setattr("dwarl.robot.config.materialize", lambda *args: None)
    monkeypatch.setattr("dwarl.datasets.cache.dataset", lambda *args: (tmp_path / "data", "dataset"))
    monkeypatch.setattr("dwarl.runs.provenance.source_identity", lambda: {})
    monkeypatch.setattr("dwarl.runs.provenance.versions", lambda: {})
    monkeypatch.setattr("dwarl.workflows.context.remember_run", lambda *args: None)
    env = SimpleNamespace(close=lambda: None)
    nav = ModuleType("dwarl.training.environment.nav_env")
    nav.NavVecEnv = lambda _: env
    monkeypatch.setitem(sys.modules, nav.__name__, nav)
    learned = []
    runner = SimpleNamespace(current_learning_iteration=0, logger=SimpleNamespace(writer=None),
                             learn=learned.append)
    monkeypatch.setattr("rsl_rl.runners.OnPolicyRunner", lambda *args, **kwargs: runner)
    monkeypatch.setattr("dwarl.training.runner_adapter.RunnerAdapter.for_runner",
                        lambda _: SimpleNamespace(after_save=[]))
    checks = SimpleNamespace(state_dict=lambda: {})
    monkeypatch.setattr("dwarl.training.validation.attach_validation", lambda *args: checks)
    monkeypatch.setattr("dwarl.training.checkpoint.attach", lambda *args: None)
    monkeypatch.setattr("dwarl.training.diagnostics.attach_diagnostics", lambda *args: None)
    monkeypatch.setattr("dwarl.training.progress.display", lambda *args: nullcontext())
    initialized = []

    def warm_start(runner, path, device):
        # The new destination is now visible to latest, but the source was frozen earlier.
        assert resolve_run("latest") != origin
        initialized.append(path)
        return {"copied": 1}

    monkeypatch.setattr("dwarl.training.checkpoint.warm_start", warm_start)
    return SimpleNamespace(cfg=cfg, origin=origin, initialized=initialized, learned=learned)


@pytest.mark.parametrize(("selector", "expected"), [("best", 10), ("latest", 20), ("10", 10)])
def test_latest_source_is_resolved_before_destination(workflow, selector, expected, capsys):
    manifest_path = workflow.origin / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["best"] = {"sha256": manifest["completed"][0]["sha256"]}
    write_json(manifest_path, manifest)
    workflow.cfg.train.init_checkpoint = selector
    destination = execute(workflow.cfg)
    assert destination != workflow.origin
    assert workflow.initialized == [workflow.origin / f"model_{expected}.pt"]
    assert workflow.learned == [workflow.cfg.train.max_iterations]
    output = capsys.readouterr()
    assert "Initialized from" in output.err
    assert "Initialized from" not in output.out


def test_absent_best_falls_back_to_latest(workflow):
    execute(workflow.cfg)
    assert workflow.initialized == [workflow.origin / "model_20.pt"]


@pytest.mark.parametrize(("damage", "error", "message"), [
    ("checkpoint", ValueError, "checkpoint missing or changed"),
    ("contract", ValueError, "saved training contract has changed"),
    ("dangling_best", ValueError, "no completed checkpoint 'best'"),
    ("schema", ValueError, "unsupported checkpoint manifest"),
    ("json", ValueError, "Expecting"),
    ("missing_file", FileNotFoundError, "model_10.pt"),
])
def test_invalid_best_does_not_fall_back_or_publish_destination(workflow, damage, error, message):
    manifest_path = workflow.origin / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["best"] = {"sha256": manifest["completed"][0]["sha256"]}
    if damage == "checkpoint":
        (workflow.origin / "model_10.pt").write_text("corrupt")
    elif damage == "contract":
        (workflow.origin / "contract.yaml").write_text("changed")
    elif damage == "dangling_best":
        manifest["best"]["sha256"] = "not-completed"
    elif damage == "schema":
        manifest["schema"] = "unsupported"
    elif damage == "missing_file":
        (workflow.origin / "model_10.pt").unlink()
    write_json(manifest_path, manifest)
    if damage == "json":
        manifest_path.write_text("{")
    with pytest.raises(error, match=message):
        execute(workflow.cfg)
    assert list(Path("runs").iterdir()) == [workflow.origin.relative_to(Path.cwd())]
    assert not workflow.initialized
    assert not workflow.learned
