import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from dwarl.config import compose_config
from dwarl.testing.comparison import paired_results
from dwarl.testing.scoring import score


def test_cli_and_web_use_the_same_comparison_option(tmp_path, monkeypatch):
    from dwarl.cli import dispatch, parser
    from dwarl.web.jobs import command_arguments

    monkeypatch.chdir(tmp_path)
    argv = ["test", "--sim", "gazebo", "--obstacle-comparison", "--show", "--print-config", "--json"]
    code, config = dispatch(parser().parse_args(argv))
    assert code == 0 and config["test"]["obstacle_comparison"] and config["test"]["show"]
    assert "--obstacle-comparison" in command_arguments(argv)


@pytest.mark.parametrize("assignment", ["test.sim=mujoco", "test.unmapped_route=previous"])
def test_comparison_rejects_incompatible_inputs(assignment):
    settings = ["test.obstacle_comparison=true", assignment]
    if assignment.startswith("test.unmapped_route"):
        settings += ["test.sim=gazebo", "test.unmapped_obstacles=3"]
    with pytest.raises(ValueError, match="comparison"):
        compose_config(assignments=settings)


@pytest.fixture
def evaluation(tmp_path, train_cfg, monkeypatch):
    from dwarl.robot.config import materialize
    from dwarl.testing.backends import gazebo
    from dwarl.workflows import test

    run = tmp_path / "run"
    run.mkdir()
    materialize(train_cfg, run / "robot")
    cfg = compose_config(assignments=["test.sim=gazebo", "test.obstacle_comparison=true",
                                     "test.controllers=[dwarl,rpp]", "test.max_cases=2",
                                     "test.suite=validation", "test.show=true"])
    checkpoint = SimpleNamespace(run=run, iteration=50, sha256="frozen", contract="contract")
    monkeypatch.setattr(test, "load_contract", lambda run: deepcopy(train_cfg))
    monkeypatch.setattr(test, "prepare_policy", lambda checkpoint: run / "bundle")
    calls, placements, closed = [], [], []
    behavior = {"mode": "success"}

    def place(md, case, count, robot, **kwargs):
        placements.append(deepcopy(case))
        return np.array([[1, *case["reference_path"][1], 0, .2, .2]])

    monkeypatch.setattr("dwarl.testing.unmapped.place_unmapped", place)

    class Backend:
        def __init__(self, contract, bundle, workload, directory, request, controller="dwarl", phase=None):
            self.phase, self.controller = phase, controller
            self.request = deepcopy(request)
            self.directory = directory

        def execute_case(self, case):
            calls.append((self.phase, self.controller, deepcopy(case), self.request))
            folder = self.directory / "gazebo"
            if self.phase:
                folder /= self.phase
            folder = folder / self.controller / case["id"]
            folder.mkdir(parents=True)
            (folder / "nav2_params.yaml").write_text("controller_server:\n  ros__parameters:\n    FollowPath: {}\n")
            if self.phase == "clean" and behavior["mode"] == "infra":
                raise RuntimeError("stack failed")
            if self.phase == "clean" and behavior["mode"] == "interrupt":
                raise KeyboardInterrupt
            row = score(case, success=behavior["mode"] != "navigation_failure", collision=False,
                        timeout=False, travelled=5, elapsed=6 if self.phase == "clean" else 8,
                        clearance=.2)
            if behavior["mode"] != "missing_plan":
                row["diagnostics"] = {"global_plans": [case["reference_path"]]}
            return row

        def close(self):
            closed.append((self.phase, self.controller))

    monkeypatch.setattr(gazebo, "GazeboBackend", Backend)
    return cfg, checkpoint, calls, placements, closed, behavior


def test_one_command_freezes_layouts_once_and_reports_both_passes(evaluation):
    from dwarl.workflows.test import execute

    cfg, checkpoint, calls, placements, closed, _ = evaluation
    out, summary = execute(cfg, checkpoint)
    assert summary["status"] == "completed" and summary["passed"]
    assert [c[:2] for c in calls] == [
        (phase, controller) for phase in ("clean", "obstructed") for controller in ("dwarl", "rpp")
        for _ in range(2)]
    assert len(placements) == 2  # once per case, not once per controller
    assert len(closed) == 4
    for index in range(2):
        assert calls[4 + index][2] == calls[6 + index][2]
    assert all(c[3].show for c in calls)
    assert all(c[3].unmapped_obstacles == 0 for c in calls[:4])
    assert all(c[3].unmapped_obstacles == 3 for c in calls[4:])
    layout = json.loads((out / "obstacles.json").read_text())
    assert layout["checkpoint"] == "frozen"
    assert all(v["placement"]["source_variant"] == "gazebo-clean" for v in layout["cases"].values())
    assert summary["obstacle_comparison"]["controllers"]["dwarl"]["successful_time_delta_s"] == 2
    report = (out / "report/README.md").read_text()
    assert "Clean versus obstructed" in report and "Frozen obstacle layouts" in report
    assert (out / "report/index.html").is_file()
    assert (out / "request_clean.yaml").is_file() and (out / "request_obstructed.yaml").is_file()
    from scripts.audit_comparison import audit

    assert audit(out)["evidence_consistent"]
    records = json.loads((out / "episodes.json").read_text())
    records["gazebo-obstructed"][0]["unmapped_obstacles"] = []
    (out / "episodes.json").write_text(json.dumps(records))
    assert any("frozen layout" in error for error in audit(out)["errors"])


@pytest.mark.parametrize("mode,status", [("missing_plan", "infrastructure_failure"),
                                        ("infra", "infrastructure_failure"), ("interrupt", "interrupted")])
def test_missing_evidence_never_starts_second_pass(evaluation, mode, status):
    from dwarl.workflows.test import execute

    cfg, checkpoint, calls, placements, closed, behavior = evaluation
    behavior["mode"] = mode
    out, summary = execute(cfg, checkpoint)
    assert summary["status"] == status and not summary["passed"]
    assert all(c[0] == "clean" for c in calls)
    assert placements == [] and closed
    assert summary["simulators"]["gazebo-obstructed"]["completed"] == 0
    assert not summary["simulators"]["gazebo-obstructed"]["complete"]
    assert (out / "summary.json").is_file()


def test_completed_navigation_failures_are_not_filtered_out(evaluation):
    from dwarl.workflows.test import execute

    cfg, checkpoint, calls, _, _, behavior = evaluation
    behavior["mode"] = "navigation_failure"
    _, summary = execute(cfg, checkpoint)
    assert len(calls) == 8 and summary["status"] == "completed" and not summary["passed"]
    assert summary["obstacle_comparison"]["controllers"]["dwarl"]["paired_completed"] == 2


def test_default_evaluation_is_still_single_pass(evaluation):
    from dwarl.workflows.test import execute

    cfg, checkpoint, calls, placements, _, _ = evaluation
    cfg.test.obstacle_comparison = False
    _, summary = execute(cfg, checkpoint)
    assert len(calls) == 4 and all(c[0] is None for c in calls)
    assert summary["obstacle_comparison"] is None and placements == []


def test_paired_time_does_not_compare_different_successful_cases():
    def row(key, success, elapsed, status="completed"):
        return dict(case_id=key, status=status, success=success, collision=False, completion_time=elapsed)
    rows = {"gazebo-clean": [row("a", True, 5), row("b", True, 100), row("c", False, 10)],
            "gazebo-obstructed": [row("a", True, 8), row("b", False, 20), row("c", True, 200)]}
    stats = paired_results(rows, ["dwarl"])["dwarl"]
    assert stats["paired_completed"] == 3 and stats["paired_successful"] == 1
    assert stats["successful_time_delta_s"] == 3
    assert stats["success_rate_delta"] == 0


@pytest.mark.parametrize("phase", [None, "clean", "obstructed"])
def test_gazebo_consumes_frozen_geometry_without_resampling(tmp_path, train_cfg, monkeypatch, phase):
    from dwarl.testing.backends import gazebo
    from dwarl.training.environment.dataset import MapDataset

    md = MapDataset(train_cfg.dataset.path, "val").maps[0]
    md.save(tmp_path / "map.npz")
    case = {"id": "case", "map_file": "map.npz", "time_budget": 10,
            "unmapped_obstacles": [[1, 1, 2, 0, .2, .2]],
            "unmapped_placement": {"route": "nav2_first_plan", "source": "clean"}}
    backend = gazebo.GazeboBackend.__new__(gazebo.GazeboBackend)
    backend.directory, backend.phase, backend.controller = tmp_path, phase, "rpp"
    backend.contract, backend.bundle, backend.domain = train_cfg, tmp_path / "bundle", 120
    backend.request = compose_config(assignments=["test.sim=gazebo", "test.unmapped_obstacles=3"]).test
    backend._routes = None
    destination = tmp_path / "gazebo"
    if phase:
        destination /= phase
    destination /= "rpp/case"
    def process(*args, **kwargs):
        from dwarl.testing.backends.ros_worker import load_case_map

        loaded = load_case_map(destination, json.loads((destination / "input.json").read_text()))
        np.testing.assert_array_equal(loaded.primitives, md.primitives)
        (destination / "result.json").write_text(json.dumps({"status": "completed"}))
        return SimpleNamespace(pid=123, returncode=0, wait=lambda **kw: None)
    monkeypatch.setattr(gazebo.subprocess, "Popen", process)
    monkeypatch.setattr(gazebo, "stop_owned", lambda *args: None)
    monkeypatch.setattr(gazebo, "lease_directory", lambda: tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("frozen obstacles must not be resampled")
    monkeypatch.setattr("dwarl.testing.unmapped.place_unmapped", forbidden)
    assert backend.execute_case(case)["status"] == "completed"
    recorded = json.loads((destination / "input.json").read_text())
    assert recorded["unmapped_obstacles"] == case["unmapped_obstacles"]
    assert recorded["unmapped_placement"] == case["unmapped_placement"]
    assert recorded["request"]["show"] is False
