"""A smoothness reward that sees weaving in window action mode."""

from types import SimpleNamespace

import numpy as np

from dwarl.navigation.rewards import angular_effort


def test_angular_effort_penalises_full_angular_acceleration():
    env = SimpleNamespace(alpha_max=2.0, control_dt=0.1, action_manager=SimpleNamespace(
        get_term=lambda name: SimpleNamespace(command_delta=np.array([[0.1, 0.2], [0.0, 0.0], [0.0, -0.1]]))))
    assert np.allclose(angular_effort(env), [-1.0, 0.0, -0.25])
