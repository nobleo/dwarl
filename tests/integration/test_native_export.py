import numpy as np
import pytest
import yaml
from tests.support import TESTBOT, WORKSPACE, policy_runner

from dwarl.config.managers import ObservationTermCfg
from dwarl.testing.scenarios import SPLIT


@pytest.mark.parametrize("multipoint", [False, True])
@pytest.mark.parametrize("command_feedback", [False, True])
def test_export_contract_and_native_parity(
        train_cfg, scenario_dir, tmp_path, multipoint, command_feedback):
    import os
    import subprocess
    from pathlib import Path

    from dwarl.artifacts.deploy_spec import DeploySpec
    from dwarl.artifacts.exporter import dump_golden, export_runner

    cfg = train_cfg.copy()
    # Compare policy commands before the external collision monitor intervenes.
    cfg.env.safety_layer.enabled = False
    cfg.observations.state.terms.command_feedback = (
        ObservationTermCfg(func="dwarl.navigation.observations:last_command", params={"action_name": "drive"})
        if command_feedback else None)
    if multipoint:
        cfg.commands.path.params.lookahead_distances = [.5, 1., 2., 3.]
    else:
        cfg.commands.path.params.lookahead_distances = [2.]
    cfg.robot.profile = str(TESTBOT)
    env, runner = policy_runner(cfg, scenario_dir, SPLIT)
    out = export_runner(runner, cfg, env, tmp_path / "bundle", iteration=0)
    assert (out / "dwarl_robot.yaml").is_file()
    n = dump_golden(env, runner.get_inference_policy(), out / "golden.npz", 40)
    g = np.load(out / "golden.npz")
    s = DeploySpec.load(out)
    expected_dim = 3 * len(s.lookahead_distances) + 4 + 2 * command_feedback
    assert s.command_feedback is command_feedback and s.obs_dim == expected_dim
    assert g["obs_state"].shape == (n, expected_dim)
    assert s.raw["version"] == (5 if command_feedback else 3)
    assert s.raw["state"].get("command_feedback", False) is command_feedback
    assert s.raw["state"]["layout"][-2:] == (
        ["last_command_v_norm", "last_command_w_norm"] if command_feedback else ["v_norm", "w_norm"])
    if command_feedback:
        assert s.raw["goal"]["tracker"] == "segment_window_v1"
        assert np.all(g["obs_state"][0, -2:] == 0)  # dump_golden starts with a reset
    else:
        assert "tracker" not in s.raw["goal"]
    if multipoint:
        assert s.lookahead_distances == [.5, 1., 2., 3.]
        assert s.lookahead_distance == 2.
    assert s.origin == pytest.approx((0.12, 0.0)) and not s.full_circle and s.raw["lidar"]["mount"]["rpy"][0] != 0
    # v4 belonged to the rejected experiment; v5 is only the explicitly marked feedback extension.
    from copy import deepcopy

    def reject(raw, match):
        path = tmp_path / "invalid"
        path.mkdir(exist_ok=True)
        (path / "deploy.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
        with pytest.raises(ValueError, match=match):
            DeploySpec.load(path)

    bad = deepcopy(s.raw)
    bad["version"] = 4
    reject(bad, "unsupported deploy.yaml version 4")
    if command_feedback:
        bad = deepcopy(s.raw)
        del bad["goal"]["tracker"]
        reject(bad, "v5 requires segment_window_v1")
        bad = deepcopy(s.raw)
        bad["state"]["command_feedback"] = False
        reject(bad, "v5 requires segment_window_v1")
    else:
        bad = deepcopy(s.raw)
        bad["state"]["command_feedback"] = True
        reject(bad, "command feedback requires deploy version 5")

    default_binary = WORKSPACE / "ros/build/dwarl_controller/test_policy_core"
    cpp = Path(os.environ.get("DWARL_TEST_CORE_BINARY", default_binary))
    if cpp.exists() or "DWARL_TEST_CORE_BINARY" in os.environ:
        result = subprocess.run([str(cpp), "--gtest_filter=Golden.ReproducesTraining"],
                                env=dict(os.environ, DWARL_TEST_EXPORT_DIR=str(out)), check=True,
                                capture_output=True, text=True)
        assert "[  PASSED  ] 1 test" in result.stdout, result.stdout
    env.close()
