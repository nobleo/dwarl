"""Robot profiles (dwarl_robot.yaml -> MuJoCo) and their use in training."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from dwarl.robot.build import build_xml, summarize
from dwarl.robot.config import materialize
from dwarl.robot.profile import ProfileError, find_package, load_profile, resolve_uri

TESTBOT = Path(__file__).parent / "robots" / "testbot"
CODY_SRC = Path(__file__).parents[2] / "cody-dwarl2" / "src"


def test_builtin_profile_resolves():
    p = load_profile("diffbot")
    assert p.id == "diffbot" and p.footprint_radius == pytest.approx(0.18) and p.lidar["num_rays"] == 180


def test_unknown_profile_lists_builtins():
    with pytest.raises(ProfileError, match="diffbot"):
        load_profile("no_such_robot")


def test_urdf_profile_geometry_and_upside_down_lidar():
    p = load_profile(TESTBOT)
    s = summarize(p, build_xml(p))
    assert s.wheel_radius == pytest.approx(0.05) and s.wheel_separation == pytest.approx(0.28)
    assert s.lidar_pos == pytest.approx([0.12, 0.0, 0.15])
    assert s.lidar_upside_down and abs(abs(s.lidar_rpy[0]) - np.pi) < 1e-6
    assert s.spawn_height == pytest.approx(0.05)
    assert p.footprint_radius == pytest.approx(np.hypot(0.21, 0.16))


def test_urdf_with_ros2_control_and_gazebo_blocks(tmp_path):
    """<joint> entries inside <ros2_control>/<gazebo> are not kinematic joints."""
    import shutil

    shutil.copytree(TESTBOT, tmp_path / "bot")
    urdf = tmp_path / "bot" / "testbot.urdf"
    blocks = ('<ros2_control name="sim" type="system"><hardware><plugin>x/Y</plugin></hardware>'
              '<joint name="left_joint"><command_interface name="velocity"/></joint>'
              '<joint name="right_joint"><command_interface name="velocity"/></joint></ros2_control>'
              '<gazebo reference="left_joint"><joint name="left_joint"/></gazebo>')
    text = urdf.read_text()
    head = text.index(">", text.index("<robot")) + 1
    urdf.write_text(text[:head] + blocks + text[head:])  # before the real joints
    expected = summarize(load_profile(TESTBOT), build_xml(load_profile(TESTBOT)))
    p = load_profile(tmp_path / "bot")
    assert summarize(p, build_xml(p)) == expected


def test_urdf_robot_drives(tmp_path):
    from dwarl.simulation.world import TiledWorld

    p = load_profile(TESTBOT)
    (tmp_path / "r.xml").write_text(build_xml(p))
    w = TiledWorld(1, pool_size=2, robot_xml=tmp_path / "r.xml")
    assert w.robot.lidar_offset == pytest.approx([0.12, 0.0, 0.15])
    w.set_robot_pose(np.array([0]), np.zeros((1, 2)), np.zeros(1))
    w.forward()
    for _ in range(400):
        w.set_ctrl(np.array([[8.0, 8.0]]))  # 0.4 m/s
        w.step(1)
    xy, yaw = w.robot_pose()
    v, om = w.robot_velocity(yaw)
    assert v[0] == pytest.approx(0.4, abs=0.04) and abs(om[0]) < 0.05 and w.robot_upright()[0] > 0.99
    assert xy[0, 0] > 0.5


def test_materialize_fills_null_keys_and_keeps_overrides(train_cfg, tmp_path):
    cfg = train_cfg.copy()
    cfg.robot.profile = str(TESTBOT)
    cfg.robot.v_max = 0.4  # explicit override wins
    info = materialize(cfg, tmp_path)
    assert cfg.robot.id == "testbot" and cfg.robot.v_max == 0.4 and cfg.robot.w_max == 1.5
    assert cfg.lidar.num_rays == 90 and cfg.lidar.noise_std == 0  # conftest pins noise to 0
    assert cfg.agent.actor.encoder_cfg.circular is False  # 240 deg scan: no wrap-around padding
    assert Path(cfg.robot.xml).is_file() and (tmp_path / "dwarl_robot.yaml").is_file()
    assert info["lidar_upside_down"]
    OmegaConf.to_yaml(cfg)  # serialisable


def test_env_casts_rays_from_the_lidar_origin(train_cfg):
    from dwarl.training.environment.nav_env import NavVecEnv

    cfg = train_cfg.copy()
    cfg.robot.profile = str(TESTBOT)
    env = NavVecEnv(cfg)
    try:
        assert env.lidar_origin == pytest.approx([0.12, 0.0])
        assert env.radius == pytest.approx(np.hypot(0.21, 0.16))
        obs = env.get_observations()
        assert obs["lidar"].shape[-1] == 90
        env.step(env.get_observations()["state"][:, :2] * 0)
    finally:
        env.close()


def test_resolve_package_uri_from_source_tree(tmp_path):
    pkg = tmp_path / "ws" / "src" / "my_robot_description"
    (pkg / "urdf").mkdir(parents=True)
    (pkg / "package.xml").write_text("<package><name>my_robot_description</name></package>")
    hints = [tmp_path / "ws" / "src"]
    assert find_package("my_robot_description", hints) == pkg
    assert resolve_uri("package://my_robot_description/urdf/r.urdf", tmp_path, hints) == pkg / "urdf" / "r.urdf"
    assert resolve_uri("$(find my_robot_description)/urdf/r.urdf", tmp_path, hints) == pkg / "urdf" / "r.urdf"


@pytest.mark.skipif(not (CODY_SRC / "pkgs" / "cody_dwarl2").is_dir(), reason="cody repo not checked out next to DWARL")
def test_cody_profile(monkeypatch):
    monkeypatch.setenv("DWARL_ROBOT_PATH", str(CODY_SRC) + os.pathsep + os.environ.get("DWARL_ROBOT_PATH", ""))
    try:
        p = load_profile("cody_dwarl2")
        s = summarize(p, build_xml(p))
    except ProfileError as e:  # no xacro available
        if str(e).startswith("xacro not found:"):
            pytest.skip(str(e))
        raise
    assert s.wheel_radius == pytest.approx(0.14) and s.wheel_separation == pytest.approx(0.51)
    assert s.lidar_pos == pytest.approx([0.43, 0.0, 0.21]) and s.lidar_upside_down
    assert p.ros["cmd_vel_stamped"] is True
