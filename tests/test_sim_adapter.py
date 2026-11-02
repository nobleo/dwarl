"""The reference adapter must override upstream entity defaults for ground-truth scoring."""

import importlib.util
from pathlib import Path

import pytest


def test_reference_adapter_forwards_nonzero_pose_and_entity(tmp_path, monkeypatch):
    launch = pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    path = Path(__file__).resolve().parents[1] / "ros/src/pkgs/dwarl_bringup/launch/tb3_sim.launch.py"
    spec = importlib.util.spec_from_file_location("tb3_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "urdf").mkdir()
    (tmp_path / "urdf/turtlebot3_waffle.urdf").write_text('<robot name="test"/>')
    monkeypatch.setattr(module, "get_package_share_directory", lambda _: str(tmp_path))
    context = launch.LaunchContext()
    context.launch_configurations.update(world="case.sdf", x="1.0", y="2.0", yaw="0.4", gui="false")
    actions = module._setup(context)
    arguments = dict(actions[1].launch_arguments)
    assert arguments == {"robot_name": "robot", "x_pose": "1.0", "y_pose": "2.0", "yaw": "0.4"}
