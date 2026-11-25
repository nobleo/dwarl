"""A real, bounded new-user journey: model -> train -> resume -> ONNX test -> installed bundle."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from tests.support import TESTBOT


def test_new_robot_journey(tmp_path):
    def run(*args, expected=(0,), logs=None):
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NO_COLOR": "1"}
        result = subprocess.run([sys.executable, "-m", "dwarl", *map(str, args), "--json"],
                                cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180)
        assert result.returncode in expected, result.stdout + result.stderr
        if logs is not None:
            logs.extend(result.stderr.splitlines())
        return json.loads(result.stdout)

    profile = TESTBOT / "dwarl_robot.yaml"
    integration = tmp_path / "robot"
    created = run("robot", "init", integration, "--answers", profile, "--profile-only", "--yes")
    assert Path(created["profile"]).is_file()
    run("setup", "--robot", integration)
    assert not (tmp_path / ".dwarl/datasets").exists()
    logs = []
    trained = run("train", "--preset", "smoke", logs=logs)
    training = Path(trained["run"])
    command = shlex.split(next(line.split("ahead of time, run: ")[1] for line in logs if "ahead of time, run:" in line))
    assert command[:2] == ["dwarl", "datagen"]
    prepared = run(*command[1:])
    assert json.loads((training / "provenance.json").read_text())["dataset"] == prepared["index_sha256"]
    assert trained["state"] == "finished"
    assert (training / "training.log").stat().st_size
    assert json.loads((training / "progress.json").read_text())["target"] == 2
    resumed = run("train", "--resume", training, "--max-iterations", 4)
    assert resumed["state"] == "finished"
    tested = run("test", "--run", training, "--case", "doorway", "--set", "test.max_cases=1", expected=(0, 1))
    assert tested["status"] == "completed", tested
    report = run("report", tested["directory"])
    assert Path(report["html"]).is_file()
    installed = run("deploy", integration, "--run", training)
    assert (Path(installed["bundle"]) / "policy.onnx").is_file()
    status = run("status")
    assert status["installed"]["checkpoint"] == tested["checkpoint"]
    assert status["next"][-1] == "navigation"  # smoke is never presented as a navigation-quality policy
