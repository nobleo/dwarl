from types import SimpleNamespace

import pytest

from dwarl import ros_env
from dwarl.workflows import setup


def _integration(ws, package, nested="pkgs"):
    pkg = ws / "src" / nested / package
    pkg.mkdir(parents=True)
    (pkg / "package.xml").write_text(f"<package><name>{package}</name></package>")
    (pkg / "dwarl_robot.yaml").write_text("schema: dwarl_robot/1\n")
    return pkg


def test_discovers_integrations_and_skips_artifacts(tmp_path):
    ws = tmp_path / "robot"
    _integration(ws, "my_robot_dwarl")
    policy = ws / "src/pkgs/my_robot_dwarl/policies/old"
    policy.mkdir(parents=True)
    (policy / "dwarl_robot.yaml").write_text("frozen copy\n")
    assert [name for name, _ in setup.integrations(ws)] == ["my_robot_dwarl"]


def test_saved_answers_round_trip_with_spaces(tmp_path):
    path = tmp_path / ".dwarl/setup.env"
    setup.save({"DWARL_ROBOT_WS": "/a b/robot", "DWARL_ROBOT_PACKAGES": "one two", "EMPTY": ""}, path)
    assert setup.load_saved(path) == {"DWARL_ROBOT_WS": "/a b/robot", "DWARL_ROBOT_PACKAGES": "one two"}


def test_non_interactive_setup_detects_sibling_and_saves(tmp_path, monkeypatch):
    dwarl, ws = tmp_path / "dwarl", tmp_path / "robot"
    dwarl.mkdir()
    _integration(ws, "my_robot_dwarl")
    monkeypatch.chdir(dwarl)
    monkeypatch.delenv("DWARL_ROBOT_WS", raising=False)
    assert setup.execute(interactive=False, build=False) == 0
    saved = setup.load_saved(dwarl / ".dwarl/setup.env")
    assert saved == {"DWARL_ROBOT_WS": str(ws.resolve()), "DWARL_ROBOT_PACKAGES": "my_robot_dwarl"}


def test_only_unleased_dwarl_sessions_are_orphans(tmp_path, monkeypatch):
    import fcntl

    monkeypatch.setattr(ros_env, "lease_directory", lambda: tmp_path)
    (tmp_path / "120.session").write_text("1000")  # test finished: lease free
    (tmp_path / "121.session").write_text("2000")  # test running: lease held
    held = (tmp_path / "121").open("a")
    fcntl.flock(held, fcntl.LOCK_EX)
    servers = {"11": 1000, "22": 2000, "33": 3000}  # 33: a Gazebo the user started
    monkeypatch.setattr(ros_env.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=" ".join(servers)))
    monkeypatch.setattr(ros_env.os, "getsid", lambda pid: servers[str(pid)])
    assert ros_env.orphaned_gazebo() == [("11", 120)]


def test_reports_missing_ros_dependencies_through_workspace_packages(tmp_path):
    ws = tmp_path / "robot"
    for name, deps in {"my_robot_dwarl": ["my_robot_sim", "rclpy", "python3-yaml"],
                       "my_robot_sim": ["twist_mux", "ros_gz_sim", "boost"],
                       "my_robot_hardware": ["vendor_can_driver"]}.items():
        pkg = ws / "src" / name
        pkg.mkdir(parents=True)
        tags = "".join(f"<exec_depend>{d}</exec_depend>" for d in deps)
        (pkg / "package.xml").write_text(f"<package><name>{name}</name>{tags}</package>")
    prefix = tmp_path / "env"
    for installed in ("rclpy", "ros_gz_sim"):
        marker = prefix / "share/ament_index/resource_index/packages" / installed
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    # hardware-only vendor_can_driver is outside the integration's closure; system keys are ignored
    assert setup.missing_ros_dependencies(ws, ["my_robot_dwarl"], [str(prefix)]) == ["twist_mux"]


def test_build_and_simulator_lifetimes_cannot_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(ros_env, "lease_directory", lambda: tmp_path)
    with ros_env.acquire_stack_lock():
        with pytest.raises(ValueError, match="active"):
            ros_env.acquire_stack_lock(build=True)
    with ros_env.acquire_stack_lock(build=True):
        with pytest.raises(ValueError, match="active"):
            ros_env.acquire_stack_lock()


def test_overlay_refreshes_when_an_existing_workspace_gains_a_package(tmp_path, monkeypatch):
    install = tmp_path / ".pixi/robot/install"
    install.mkdir(parents=True)
    (install / "local_setup.sh").touch()
    monkeypatch.setattr(ros_env, "root", lambda: tmp_path)
    monkeypatch.setattr(ros_env, "load_setup", lambda: None)
    monkeypatch.setenv("AMENT_PREFIX_PATH", str(install / "old_package"))
    new_prefix = f"{install}/new_package:{install}/old_package"
    monkeypatch.setattr(ros_env.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=f"AMENT_PREFIX_PATH={new_prefix}\0".encode()))
    ros_env.overlay()
    import os

    assert os.environ["AMENT_PREFIX_PATH"] == new_prefix
