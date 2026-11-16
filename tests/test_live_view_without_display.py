"""The live view of a running training must never end the training when no window can be opened."""

import importlib.util
import warnings
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_no_window_raises_instead_of_launching(monkeypatch):
    from dwarl.simulation import viewer

    launch = Mock()
    monkeypatch.setattr(viewer, "display_available", lambda: False)
    monkeypatch.setattr(viewer.mujoco.viewer, "launch_passive", launch)
    with pytest.raises(RuntimeError, match="cannot open a window"):
        viewer.SimViewer(SimpleNamespace())
    launch.assert_not_called()  # MuJoCo would end the process if GLFW fails inside it


@pytest.mark.skipif(importlib.util.find_spec("glfw") is None, reason="glfw not installed")
def test_display_check_survives_a_missing_display(monkeypatch):
    from dwarl.simulation.viewer import display_available

    monkeypatch.setenv("DISPLAY", "")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert display_available() is False      # the check fails in the child; this process carries on


def test_toggle_without_display_keeps_training(tmp_path, monkeypatch):
    from dwarl.simulation import viewer
    from dwarl.training.environment.nav_env import NavVecEnv

    def no_window(*args, **kwargs):
        raise RuntimeError("cannot open a window on this display")

    monkeypatch.setattr(viewer, "SimViewer", no_window)
    trigger = tmp_path / "view"
    trigger.touch()
    env = SimpleNamespace(viewer=None, viewer_trigger=trigger, _viewer_poll=0.0, world=SimpleNamespace(num_envs=4),
                          cfg=SimpleNamespace(sim=SimpleNamespace(render_fps=30.0)), _episode_paths={}, num_envs=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        NavVecEnv._poll_viewer(env)
    assert env.viewer is None and not trigger.exists()   # toggle reset, training continues headless
    assert any("live view unavailable" in str(w.message) for w in caught)
