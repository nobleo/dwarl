import json
from types import SimpleNamespace

from dwarl.config import compose_config
from dwarl.runs.store import sha256, write_json
from dwarl.training.runner_adapter import RunnerAdapter
from dwarl.training.validation import attach_validation


def test_best_ignores_held_out_report_even_if_newer(tmp_path, monkeypatch):
    run = tmp_path
    checkpoint = run / "weights.pt"
    checkpoint.write_bytes(b"complete")
    write_json(run / "checkpoint_manifest.json", {"completed": [], "best": None})
    cfg = compose_config(cwd=tmp_path)
    cfg.train.validation_interval = 1
    runner = SimpleNamespace(save=lambda *args: None, current_learning_iteration=1)

    def evaluate(*args, **kwargs):
        report = {"iteration": 1, "checkpoint": sha256(checkpoint), "suite": "validation",
                  "status": "completed", "workload": "validation-workload",
                  "simulators": {"mujoco": {"complete": True, "success_rate": .25}}}
        write_json(run / "tests/validation/summary.json", report)
        report["suite"] = "test"
        report["simulators"]["mujoco"]["success_rate"] = 1.
        write_json(run / "tests/held-out/summary.json", report)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("dwarl.training.validation.subprocess.run", evaluate)
    attach_validation(runner, run, cfg)
    RunnerAdapter.for_runner(runner).after_save[0](checkpoint)
    best = json.loads((run / "checkpoint_manifest.json").read_text())["best"]
    assert best["value"] == .25
    assert best["evidence"] == "tests/validation/summary.json"
