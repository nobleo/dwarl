import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dwarl.config import compose_config
from dwarl.runs.store import sha256, write_json
from dwarl.training.runner_adapter import RunnerAdapter
from dwarl.training.validation import EarlyStop, TrainingChecks, attach_validation


def setup_checks(tmp_path, monkeypatch, *, patience=2):
    cfg = compose_config(cwd=tmp_path, assignments=[
        "train.save_interval=1", "train.validation_interval=1", "train.benchmark_interval=1",
        "train.early_stop.enabled=true", f"train.early_stop.patience={patience}",
        "train.early_stop.min_delta=0.1"])
    write_json(tmp_path / "checkpoint_manifest.json", {"completed": [], "best": None})
    scalars = {}
    writer = SimpleNamespace(add_scalar=lambda tag, value, step: scalars.update({(tag, step): value}),
                             flush=lambda: None)

    def save(path, infos):
        Path(path).write_text(json.dumps(infos))

    runner = SimpleNamespace(save=save, current_learning_iteration=1)
    checks = attach_validation(runner, tmp_path, cfg)
    runner.logger = SimpleNamespace(writer=writer)
    RunnerAdapter.for_runner(runner).checkpoint_state.append(lambda: {"dwarl_training": checks.state_dict()})
    results = {"validation": .5, "benchmark": 1.}
    calls = []

    def evaluate(command, **kwargs):
        suite = command[command.index("--suite") + 1]
        calls.append((suite, runner.current_learning_iteration))
        checkpoints = sorted(tmp_path.glob("model*.pt"), key=lambda p: p.stat().st_mtime_ns)
        score = results[suite]
        if score == "failure":
            return SimpleNamespace(returncode=2)
        completed = score != "partial"
        report = {"iteration": runner.current_learning_iteration, "checkpoint": sha256(checkpoints[-1]),
                  "suite": suite, "status": "completed", "workload": f"{suite}-fixed",
                  "simulators": {"mujoco": {"complete": completed, "success_rate": score if completed else .5,
                                           "collision_rate": .1, "expected": 2, "completed": 2 if completed else 1}}}
        write_json(tmp_path / "tests" / str(len(calls)) / "summary.json", report)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("dwarl.training.validation.subprocess.run", evaluate)
    return cfg, runner, checks, results, calls, scalars


def test_checks_disabled_by_default(tmp_path, monkeypatch):
    cfg = compose_config(cwd=tmp_path)
    assert cfg.train.validation_interval == cfg.train.benchmark_interval == 0
    assert not cfg.train.early_stop.enabled
    runner = SimpleNamespace(current_learning_iteration=200)
    checks = TrainingChecks(runner, tmp_path, cfg)
    monkeypatch.setattr(checks, "evaluate", lambda *args: pytest.fail("disabled check ran"))
    checks.after_save(tmp_path / "unused")
    assert not (tmp_path / "training_checks").exists()


@pytest.mark.parametrize("assignments,match", [
    (["train.early_stop.enabled=true"], "requires"),
    (["train.validation_interval=7"], "multiple"),
    (["train.benchmark_interval=-1"], "nonnegative"),
    (["train.early_stop.patience=0"], "positive"),
    (["train.early_stop.min_delta=-0.1"], "between"),
    (["train.benchmark_cases=0"], "positive"),
])
def test_invalid_check_settings(tmp_path, assignments, match):
    with pytest.raises(ValueError, match=match):
        compose_config(cwd=tmp_path, assignments=assignments)


def test_plateau_stops_only_after_completed_validation_and_logs(tmp_path, monkeypatch):
    cfg, runner, checks, results, calls, scalars = setup_checks(tmp_path, monkeypatch)
    runner.save(str(tmp_path / "model.pt"))
    results["validation"] = .55  # best checkpoint improves, but not by the early-stop delta
    runner.current_learning_iteration = 2
    runner.save(str(tmp_path / "model.pt"))
    assert checks.state["bad_checks"] == 1
    assert json.loads((tmp_path / "checkpoint_manifest.json").read_text())["best"]["value"] == .55
    for iteration, result in ((3, "failure"), (4, "partial")):
        runner.current_learning_iteration = iteration
        results["validation"] = result
        runner.save(str(tmp_path / "model.pt"))
        assert checks.state["bad_checks"] == 1
    runner.current_learning_iteration = 5
    results["validation"] = .54
    with pytest.raises(EarlyStop, match="2 completed checks"):
        runner.save(str(tmp_path / "model.pt"))
    assert scalars["EarlyStop/patience_remaining", 5] == 0
    assert scalars["EarlyStop/stopped", 5] == 1
    assert scalars["Validation/best_iteration", 5] == 2
    assert scalars["Benchmark/success_rate", 5] == 1.
    assert scalars["Validation/complete", 3] == scalars["Validation/complete", 4] == 0


def test_resume_uses_post_check_sidecar_and_does_not_repeat_checks(tmp_path, monkeypatch):
    cfg, runner, checks, results, calls, _ = setup_checks(tmp_path, monkeypatch)
    runner.save(str(tmp_path / "model.pt"))
    runner.current_learning_iteration = 2
    runner.save(str(tmp_path / "model.pt"))
    path = sorted(tmp_path.glob("model*.pt"), key=lambda p: p.stat().st_mtime_ns)[-1]
    infos = json.loads(path.read_text())
    assert infos["dwarl_training"]["bad_checks"] == 0  # saved before check
    resumed = TrainingChecks(runner, tmp_path, cfg)
    resumed.restore(SimpleNamespace(sha256=sha256(path)), infos)
    assert resumed.state["bad_checks"] == 1  # sidecar includes check
    resumed.after_save(path)
    assert len(calls) == 4
    runner.current_learning_iteration = 3
    with pytest.raises(EarlyStop):
        resumed.after_save(path)


def test_resume_fallback_and_old_checkpoint(tmp_path):
    cfg = compose_config(cwd=tmp_path)
    checks = TrainingChecks(SimpleNamespace(), tmp_path, cfg)
    checkpoint = SimpleNamespace(sha256="missing")
    checks.restore(checkpoint, {})
    assert checks.state["bad_checks"] == 0
    state = checks.state_dict()
    state["bad_checks"] = 2
    checks.restore(checkpoint, {"dwarl_training": state})
    assert checks.state["bad_checks"] == 2


def test_significant_improvement_resets_patience_and_ties_do_not(tmp_path):
    cfg = compose_config(cwd=tmp_path)
    cfg.train.early_stop.enabled = True
    cfg.train.early_stop.min_delta = .1
    checks = TrainingChecks(SimpleNamespace(), tmp_path, cfg)
    for iteration, value in enumerate((.5, .55, .7), 1):
        checks.observe_validation(value, iteration)
    assert checks.state["bad_checks"] == 0
    assert checks.state["best_iteration"] == 3
    cfg.train.early_stop.min_delta = 0
    checks.observe_validation(.7, 4)
    assert checks.state["bad_checks"] == 1


def test_final_save_deduplicates_and_benchmark_never_selects_best(tmp_path, monkeypatch):
    cfg, runner, checks, results, calls, _ = setup_checks(tmp_path, monkeypatch)
    cfg.train.early_stop.enabled = False
    cfg.train.validation_interval = 0
    runner.save(str(tmp_path / "model.pt"))
    runner.save(str(tmp_path / "model.pt"))
    assert calls == [("benchmark", 1)]
    assert checks.state["bad_checks"] == 0
    assert json.loads((tmp_path / "checkpoint_manifest.json").read_text())["best"] is None
