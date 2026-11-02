"""User journeys and failure boundaries, rather than terminal-format snapshots."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from dwarl.cli import dispatch, parser, ui
from dwarl.config import compose_config
from dwarl.robot.init import execute as initialize
from dwarl.robot.init import questionnaire, simulation_wrapper
from dwarl.robot.profile import load_profile
from dwarl.workflows import context

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "tests/robots/testbot/dwarl_robot.yaml"


def cli(tmp_path, *args):
    environment = {**os.environ, "NO_COLOR": "1"}
    return subprocess.run([sys.executable, "-m", "dwarl", *args], cwd=tmp_path, env=environment,
                          capture_output=True, text=True, timeout=60)


def test_generated_package_compiles_and_preserves_edited_launch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "robot/src/testbot_dwarl"
    initialize(target, answers=PROFILE, interactive=False, maintainer="Test team", email="team@example.org")
    profile = load_profile(target)
    assert profile.uri(profile.raw["description"]["urdf"]).is_file()
    assert context.load()["robot"] == str(profile.path)
    assert "install(FILES dwarl_robot.yaml" in (target / "CMakeLists.txt").read_text()
    launch = target / "launch/navigation.launch.py"
    launch.write_text(launch.read_text() + "\n# user customization\n")
    original = (target / "dwarl_robot.yaml").read_bytes()
    with pytest.raises(ValueError, match="Preserving user-edited"):
        initialize(target, interactive=False, update=True, assignments=["limits.v_max=0.4"])
    assert (target / "dwarl_robot.yaml").read_bytes() == original
    assert "user customization" in launch.read_text()


def test_invalid_profile_never_publishes_integration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "robot/src/testbot_dwarl"
    with pytest.raises(ValueError, match="Model validation failed"):
        initialize(target, answers=PROFILE, interactive=False, maintainer="Team", email="team@example.org",
                   assignments=["drive.left_wheel_joint=does_not_exist"])
    assert not (target / "dwarl_robot.yaml").exists()
    assert list((tmp_path / ".dwarl/drafts").glob("*.json"))


def test_interactive_wizard_accepts_and_validates_supplied_measurements(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "")
    target = tmp_path / "robot/src/testbot_dwarl"
    result = initialize(target, answers=PROFILE, interactive=True, profile_only=True)
    profile = load_profile(result["profile"])
    assert profile.id == "testbot"
    assert profile.limits["v_max"] == .6
    assert profile.lidar["fov_deg"] == 240


def test_missing_noninteractive_inputs_are_reported_together(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError) as caught:
        initialize(tmp_path / "robot", interactive=False)
    assert "limits.v_max" in str(caught.value) and "lidar.fov_deg" in str(caught.value)
    assert not (tmp_path / "robot").exists()


def test_wizard_back_and_interruption_preserve_values(tmp_path, monkeypatch):
    draft = tmp_path / "draft.json"
    answers = iter(["12", ":back", "14", "9"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    data = {}
    fields = [("a", "First", None, int, ()), ("b", "Second", None, int, ())]
    questionnaire(data, fields, draft)
    assert data == json.loads(draft.read_text()) == {"a": 14, "b": 9}
    answers = iter(["4"])

    def interrupt(_):
        try:
            return next(answers)
        except StopIteration as exc:
            raise KeyboardInterrupt from exc

    monkeypatch.setattr("builtins.input", interrupt)
    with pytest.raises(KeyboardInterrupt):
        questionnaire({}, fields, draft)
    assert json.loads(draft.read_text()) == {"a": 4}


def test_presets_are_explicit_and_overridable(tmp_path):
    assert compose_config(cwd=tmp_path).train.validation_interval == 0
    smoke = compose_config(cwd=tmp_path, preset="smoke", assignments=["train.num_envs=4"])
    assert smoke.train.max_iterations == 2 and smoke.train.num_envs == 4
    assert smoke.train.preset == "smoke" and smoke.train.data.train_maps == 1
    (tmp_path / "dwarl.yaml").write_text("train:\n  validation_interval: 100\n")
    assert compose_config(cwd=tmp_path, preset="navigation").train.validation_interval == 100


def test_json_stdout_is_clean_and_errors_keep_exit_codes(tmp_path):
    checked = cli(tmp_path, "robot", "check", str(PROFILE), "--json")
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["passed"]
    assert "\x1b" not in checked.stdout + checked.stderr
    missing = cli(tmp_path, "--json", "robot", "check", "missing-robot")
    assert missing.returncode == 2
    assert not json.loads(missing.stdout)["passed"]
    empty = cli(tmp_path, "train", "--json")
    assert empty.returncode == 2
    assert "select a robot" in json.loads(empty.stdout)["error"]


def test_context_separates_robots_and_smoke_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    context.select_robot(PROFILE)
    context.remember_run(PROFILE, "runs/first", "navigation")
    context.remember_run(PROFILE, "runs/smoke", "smoke")
    assert context.selected_run() == str(tmp_path / "runs/first")
    context.select_robot("diffbot")
    assert context.selected_run() is None
    # Explicit config beats the remembered selection.
    (tmp_path / "experiment.yaml").write_text("robot: diffbot\n")
    args = parser().parse_args(["train", "--config", "experiment.yaml", "--print-config", "--json"])
    _, payload = dispatch(args)
    assert payload["robot"] == "diffbot"


def test_mujoco_cli_never_builds_ros(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from dwarl.workflows import prepare, test

    monkeypatch.setattr(prepare, "ros", lambda *a: pytest.fail("MuJoCo must not build ROS"))
    monkeypatch.setattr(test, "execute", lambda cfg: (tmp_path, {"status": "completed", "passed": False}))
    code, _ = dispatch(parser().parse_args(["test", "--sim", "mujoco", "--run", "runs/pinned"]))
    assert code == 1


def test_sim_wrapper_maps_arguments_and_rejects_ambiguous_mapping():
    mapping = {"world": "world_file", "x": "x_pose", "y": "y_pose", "yaw": "yaw", "gui": "gui"}
    script = simulation_wrapper("robot_sim", "robot.launch.py", mapping)
    compile(script, "simulation.launch.py", "exec")
    assert "'world_file': LaunchConfiguration('world')" in script
    with pytest.raises(ValueError, match="distinct"):
        simulation_wrapper("robot_sim", "robot.launch.py", {**mapping, "y": "x_pose"})


def test_report_html_is_local_readable_and_escapes_values(tmp_path):
    from dwarl.workflows.report import render_html

    report = tmp_path / "report"
    report.mkdir()
    (report / "README.md").write_text("# Results\n\n<script>alert(1)</script>\n\n![route](route.png)\n")
    result = render_html(tmp_path).read_text()
    assert '<script>' not in result and "&lt;script&gt;" in result
    assert 'src="route.png"' in result


def test_guidance_checks_evidence_not_directory_existence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from dwarl.workflows.guidance import inspect

    context.select_robot(PROFILE)
    (tmp_path / "runs/fake/policies").mkdir(parents=True)
    result = inspect()
    assert result["run"] is None and result["next"][-1] == "smoke"


def test_run_watch_json_is_a_snapshot(tmp_path, monkeypatch):
    from dwarl.workflows import guidance

    monkeypatch.setattr(guidance, "run_info", lambda _: {
        "run": str(tmp_path), "state": {"state": "running"}, "checkpoint": None,
        "progress": {}, "checkpoint_error": None})
    monkeypatch.setattr(guidance.time, "sleep", lambda _: pytest.fail("JSON watch must not sleep"))
    _, result = guidance.execute(SimpleNamespace(command="runs", action="watch", run="any", json=True, interval=1))
    assert result["state"]["state"] == "running"


def test_generated_launch_and_schema_examples_ship_as_resources():
    from importlib.resources import files

    template = files("dwarl").joinpath("templates/navigation.launch.py").read_text()
    compile(template, "navigation.launch.py", "exec")
    assert "selected_policy" in template
    example = yaml.safe_load(files("dwarl").joinpath("templates/dwarl_robot.yaml").read_text())
    assert example["schema"] == "dwarl_robot/1"


def test_setup_example_selects_builtin_and_skips_ros_build(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from dwarl.workflows import prepare, setup

    monkeypatch.setattr(prepare, "ros", lambda *a: pytest.fail("Example setup must not build ROS"))
    assert setup.execute(interactive=False, example=True) == 0
    assert context.load()["robot"].endswith("robots/diffbot/dwarl_robot.yaml")


def test_ambiguous_workspace_is_not_silently_selected(tmp_path, monkeypatch):
    from dwarl.workflows.setup import guess_workspace

    checkout = tmp_path / "dwarl"
    checkout.mkdir()
    for name in ("one", "two"):
        pkg = tmp_path / name / "src" / name
        pkg.mkdir(parents=True)
        (pkg / "package.xml").write_text(f"<package><name>{name}</name></package>")
        (pkg / "dwarl_robot.yaml").touch()
    monkeypatch.chdir(checkout)
    monkeypatch.delenv("DWARL_ROBOT_WS", raising=False)
    assert guess_workspace({}) == ""


@pytest.fixture(autouse=True)
def reset_ui():
    ui.configure(plain=True)
