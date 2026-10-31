import numpy as np
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from dwarl.config.managers import ObservationTermCfg
from dwarl.datasets.generation.mapdata import MapData
from dwarl.testing.scenarios import SCENARIOS, SPLIT, build_scenarios


@pytest.fixture(scope="module")
def scenario_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("scen")
    build_scenarios(out, previews=False)
    return out


def test_scenarios_build_and_paths_clear(scenario_dir):
    for name in SCENARIOS:
        md = MapData.load(scenario_dir / SPLIT / f"{name}.npz")
        assert md.generator == name and md.num_pairs >= 1
        for k in range(md.num_pairs):
            p = md.path(k)
            assert not md.occupancy[tuple(md.xy_to_cell(p).T)].any(), f"{name}: path through obstacle"


def _runner(cfg, dataset, split):
    from rsl_rl.runners import OnPolicyRunner

    from dwarl.training.environment.nav_env import NavVecEnv

    cfg.dataset.path, cfg.dataset.split, cfg.env.num_envs = str(dataset), split, 2
    env = NavVecEnv(cfg, split=split, sequential=True)
    runner = OnPolicyRunner(env, OmegaConf.to_container(cfg.agent, resolve=True), log_dir=None)
    # non-trivial normaliser statistics so the in-graph normalisation is actually exercised
    for _ in range(3):
        env.step(torch.rand(2, 2) * 2 - 1)
        runner.alg.actor.update_normalization(env.get_observations())
    return env, runner


def test_onnx_export_matches_torch(train_cfg, scenario_dir, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from dwarl.artifacts.exporter import OnnxPolicy, dump_golden, export_runner

    env, runner = _runner(train_cfg, scenario_dir, SPLIT)
    from copy import deepcopy
    before_history = env.observation_manager.get_term("lidar", "scan").buffer.copy()
    before_rng = deepcopy(env.lidar._rng.bit_generator.state)
    out = export_runner(runner, train_cfg, env, tmp_path / "exported", iteration=0, jit=True)
    assert np.array_equal(before_history, env.observation_manager.get_term("lidar", "scan").buffer)
    assert before_rng == env.lidar._rng.bit_generator.state
    spec = yaml.safe_load((out / "deploy.yaml").read_text())
    assert [i["name"] for i in spec["model"]["inputs"]] == ["obs", "lidar"]
    assert spec["version"] == 3 and spec["lidar"]["num_rays"] == 180
    assert spec["state"]["layout"][0] == "goal0_dist_norm"
    assert (out / "policy.pt").exists() and (out / "env.yaml").exists()
    from dwarl.robot.build import build_xml
    from dwarl.robot.profile import load_profile

    assert (out / "manifest.json").exists()
    build_xml(load_profile(out))

    sess = ort.InferenceSession(str(out / "policy.onnx"))
    assert [i.name for i in sess.get_inputs()] == ["obs", "lidar"]
    torch_policy = runner.get_inference_policy()
    onnx_policy = OnnxPolicy(out / "policy.onnx")
    jit = torch.jit.load(str(out / "policy.pt"))
    with torch.no_grad():
        for _ in range(20):
            obs = env.get_observations()
            a_t, a_o = torch_policy(obs), onnx_policy(obs)
            a_j = jit(obs["state"], [obs["lidar"]])
            assert torch.allclose(a_t, a_o, atol=1e-5) and torch.allclose(a_t, a_j, atol=1e-5)
            env.step(a_t)

    n = dump_golden(env, torch_policy, tmp_path / "golden.npz", 30)
    g = np.load(tmp_path / "golden.npz")
    h = len(spec["lidar"]["history"])
    assert n > 5 and g["obs_lidar"].shape == (n, h * 180) and g["plan"].shape[1] == 2
    # raw ranges -> normalised lidar obs, exactly as a runtime has to do it
    lc = spec["lidar"]
    ranges = np.clip(g["ranges"], lc["min_range"], lc["max_range"])
    if lc["normalize"] == "linear":
        norm = (ranges - lc["min_range"]) / (lc["max_range"] - lc["min_range"])
    else:
        norm = np.log1p(np.maximum(ranges - lc["min_range"], 0) / 0.1) / np.log1p(
            (lc["max_range"] - lc["min_range"]) / 0.1)
    history = lc["history"]
    expected_lidar = np.stack([np.stack([norm[max(t - lag, 0)] for lag in history]) for t in range(n)]).reshape(n, -1)
    assert np.allclose(expected_lidar, g["obs_lidar"], atol=1e-6)
    # action -> command mapping + rate limit from deploy.yaml reproduces the env
    ac, dt = spec["action"], spec["control"]["dt"]
    a = np.clip(g["action"], -1, 1)
    lim = np.array([ac["rate_limit"]["a_max"], ac["rate_limit"]["alpha_max"]]) * dt
    if ac["mode"] == "window":
        expected_cmd = np.stack([np.clip(g["cmd"][:, 0] + a[:, 0] * lim[0], ac["v"]["min"], ac["v"]["max"]),
                                 np.clip(g["cmd"][:, 1] + a[:, 1] * lim[1], -ac["w"]["max"], ac["w"]["max"])], -1)
    else:
        tgt = np.stack([ac["v"]["min"] + (a[:, 0] + 1) / 2 * (ac["v"]["max"] - ac["v"]["min"]),
                        a[:, 1] * ac["w"]["max"]], -1)
        expected_cmd = g["cmd"] + np.clip(tgt - g["cmd"], -lim, lim)
    assert np.allclose(expected_cmd, g["cmd_next"], atol=1e-6)


def test_benchmark_policy(train_cfg, scenario_dir, tmp_path):
    pytest.importorskip("onnxruntime")
    from dwarl.artifacts.exporter import benchmark_policy, export_runner

    env, runner = _runner(train_cfg, scenario_dir, SPLIT)
    out = export_runner(runner, train_cfg, env, tmp_path / "exported", iteration=0)

    # Benchmark with minimal iterations for test speed
    result = benchmark_policy(out, iters=10, warmup=2, threads=[0, 1])

    # Verify structure
    assert "system" in result
    assert "backends" in result
    assert result["system"]["iters"] == 10
    assert result["system"]["warmup"] == 2

    # Verify ONNX results (should be present)
    assert "onnx_default" in result["backends"] or "onnx_0t" in result["backends"]
    for _name, data in result["backends"].items():
        assert "mean_ms" in data and data["mean_ms"] > 0
        assert "median_ms" in data and data["median_ms"] > 0
        assert "p95_ms" in data
        assert "p99_ms" in data
        assert "max_ms" in data
        assert "hz" in data and data["hz"] > 0
        assert "metadata" in data

    # Verify JIT results (should be present)
    assert "jit" not in result["backends"]


@pytest.mark.parametrize("multipoint", [False, True])
@pytest.mark.parametrize("command_feedback", [False, True])
def test_runtime_reproduces_training_with_offset_upside_down_lidar(
        train_cfg, scenario_dir, tmp_path, multipoint, command_feedback):
    """dwarl_runtime (ROS-free deployment) rebuilds the training observations from raw sensor data, incl. a
    lidar mounted ahead of the axle and upside down (as on Cody), and maps actions like the env."""
    pytest.importorskip("onnxruntime")
    from pathlib import Path

    from dwarl_runtime import PathTracker, Policy, Scan, laser_matrix, lidar_observation, state_observation

    from dwarl.artifacts.exporter import dump_golden, export_runner

    cfg = train_cfg.copy()
    cfg.observations.state.terms.command_feedback = (
        ObservationTermCfg(func="dwarl.navigation.observations:last_command", params={"action_name": "drive"})
        if command_feedback else None)
    if multipoint:
        cfg.commands.path.params.lookahead_distances = [.5, 1., 2., 3.]
    else:
        cfg.commands.path.params.lookahead_distances = [2.]
    cfg.robot.profile = str(Path(__file__).parent / "robots" / "testbot")
    env, runner = _runner(cfg, scenario_dir, SPLIT)
    out = export_runner(runner, cfg, env, tmp_path / "bundle", iteration=0)
    assert (out / "dwarl_robot.yaml").is_file()
    n = dump_golden(env, runner.get_inference_policy(), out / "golden.npz", 40)
    g = np.load(out / "golden.npz")
    pol = Policy(out, expected_robot_id="testbot")
    pol.set_plan(g["plan"], new_goal=True)
    s = pol.spec
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
    with pytest.raises(ValueError, match="trained for robot"):
        Policy(out, expected_robot_id="cody")

    angles = s.angle_min + np.arange(s.num_rays) * s.angle_increment
    flipped = laser_matrix(([0.12, 0.0, 0.15], [1.0, 0.0, 0.0, 0.0]))  # roll = pi (quat xyzw)
    tracker = PathTracker()
    tracker.set_plan(g["plan"])
    for t in range(n):
        # Frequent same-goal replans must preserve history, action and command feedback.
        pol.set_plan(g["plan"])
        upright = Scan(g["ranges"][t], s.angle_min, s.angle_increment, 0.0, s.max_range)
        assert np.allclose(lidar_observation(s, upright, (0.12, 0.0, 0.0)), g["obs_lidar"][t][:s.num_rays], atol=1e-5)
        # the same scan as an upside-down sensor reports it: mirrored angles, reversed order
        mirrored = Scan(g["ranges"][t][::-1].copy(), -angles[-1], s.angle_increment, 0.0, s.max_range)
        assert np.allclose(lidar_observation(s, mirrored, flipped), g["obs_lidar"][t][:s.num_rays], atol=1e-5)
        st, _ = state_observation(s, tracker, g["pose"][t], g["vel"][t][0], g["vel"][t][1],
                                   g["last_action"][t], g["cmd"][t])
        assert np.allclose(st, g["obs_state"][t], atol=1e-4)
        if command_feedback:
            # Feedback is the command that drove the preceding physics step, not this tick's new action.
            assert np.allclose(st[-2:], g["cmd"][t] / [s.v_scale, s.w_scale], atol=1e-6)
        command = pol.step(upright, (0.12, 0.0, 0.0), g["pose"][t], g["vel"][t])
        assert np.allclose(command.action, g["action"][t], atol=1e-4)
        assert np.allclose((command.v, command.w), g["cmd_next"][t], atol=1e-5)
        if multipoint:
            assert np.allclose(command.goal, g["goal"][t], atol=1e-5)

    # v4 belonged to the rejected experiment; v5 is only the explicitly marked feedback extension.
    from copy import deepcopy

    from dwarl_runtime import DeploySpec

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

    import os
    import subprocess

    cpp = Path(__file__).parents[1] / "ros/build/dwarl_controller/test_policy_core"
    if cpp.exists():
        subprocess.run([str(cpp), "--gtest_filter=Golden.*"],
                       env=dict(os.environ, DWARL_TEST_EXPORT_DIR=str(out)), check=True, capture_output=True)
    env.close()


def test_command_feedback_appends_previous_command_and_resets(train_cfg, scenario_dir):
    from dwarl.training.environment.nav_env import NavVecEnv

    cfg = train_cfg.copy()
    cfg.dataset.path, cfg.dataset.split = str(scenario_dir), SPLIT
    cfg.env.num_envs = 1
    cfg.observations.state.terms.command_feedback = ObservationTermCfg(
        func="dwarl.navigation.observations:last_command", params={"action_name": "drive"})
    env = NavVecEnv(cfg, split=SPLIT, sequential=True)
    try:
        assert env.get_observations()["state"].shape == (1, 18)
        assert torch.equal(env.get_observations()["state"][0, -2:], torch.zeros(2))
        for action in (torch.tensor([[1., 1.]]), torch.tensor([[-1., 0.5]])):
            previous = env.drive_action.command.copy()
            obs, _, _, _ = env.step(action)
            assert not np.array_equal(previous, env.drive_action.command)
            assert np.allclose(obs["state"][0, -2:].numpy(), env.drive_action.command[0] / [env.v_max, env.w_max])
        env.reset()
        assert np.all(env.drive_action.command == 0)
        assert torch.equal(env.get_observations()["state"][0, -2:], torch.zeros(2))
    finally:
        env.close()
