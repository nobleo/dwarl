
import numpy as np
import pytest
from tests.support import TESTBOT

from dwarl.config import compose_config
from dwarl.robot.config import materialize
from dwarl.testing.cases import materialize as materialize_cases
from dwarl.testing.cases import select_cases
from dwarl.testing.scoring import score, summarize


@pytest.mark.parametrize("robot", ["diffbot", str(TESTBOT)])
def test_default_benchmark_includes_office_and_warehouse(tmp_path, monkeypatch, robot):
    from pathlib import Path

    from dwarl.testing.scenarios import BENCHMARK_NAMES, LAYOUT_VERSION
    from dwarl.training.contract import build_contract
    from dwarl.training.environment.nav_env import NavVecEnv

    if robot != "diffbot":
        robot = str(Path(robot).resolve())
    public = compose_config(assignments=[f"robot={robot}", "train.num_envs=1", "train.device=cpu"])
    contract = build_contract(public)
    materialize(contract, tmp_path / "robot")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "benchmark"
    out.mkdir()
    workload = materialize_cases(contract, public.test, out)
    assert {case["name"] for case in workload["cases"]} == set(BENCHMARK_NAMES)
    assert len(workload["cases"]) <= 32
    assert LAYOUT_VERSION > 4  # old cached workloads must not hide the new worlds
    contract.dataset.path, contract.dataset.split = str(out / "maps"), workload["split"]
    contract.sim.num_shards = 1
    env = NavVecEnv(contract, split=workload["split"], sequential=True)
    try:
        assert env.world.tile_size == 16.
        for name in ("office", "warehouse"):
            case = next(case for case in workload["cases"] if case["name"] == name)
            assert len(case["reference_path"]) > 50
            env.reset_case(case["map"], case["pair"], case["start"][2], case["seed"])
            xy, _ = env.world.robot_pose()
            np.testing.assert_allclose(xy[0], case["start"][:2], atol=1e-6)
    finally:
        env.close()
    public.test.case = "office"
    selected = tmp_path / "office"
    selected.mkdir()
    office = materialize_cases(contract, public.test, selected)
    assert len(office["cases"]) == 2
    assert {case["name"] for case in office["cases"]} == {"office"}
    assert {case["id"] for case in office["cases"]} == {
        case["id"] for case in workload["cases"] if case["name"] == "office"}
    target = office["cases"][0]
    public.test.dataset = str(selected / "maps")
    public.test.case_id = target["id"]
    assert materialize_cases(contract, public.test, tmp_path / "replay")["cases"] == [target]


def test_capped_cases_cover_types_and_maps_reproducibly():
    from collections import Counter

    cases = [{"name": name, "map": mi, "pair": pi}
             for name, maps in (("rooms", range(4)), ("corridors", range(4, 6)))
             for mi in maps for pi in range(32)]
    selected = select_cases(cases, 32, 42)
    assert Counter(c["name"] for c in selected) == {"rooms": 16, "corridors": 16}
    assert Counter(c["map"] for c in selected) == {0: 4, 1: 4, 2: 4, 3: 4, 4: 8, 5: 8}
    assert len({(c["map"], c["pair"]) for c in selected}) == 32
    assert selected == select_cases(cases, 32, 42)
    assert selected != select_cases(cases, 32, 43)
    assert select_cases(cases, None, 42) == cases
    assert select_cases(cases, len(cases), 42) == cases


def test_office_selection_does_not_build_unselected_layouts(tmp_path, monkeypatch):
    from dwarl.testing import scenarios
    from dwarl.training.contract import build_contract

    public = compose_config(assignments=["robot=diffbot", "test.case=office"])
    contract = build_contract(public)
    materialize(contract, tmp_path / "robot")
    monkeypatch.chdir(tmp_path)
    build = scenarios.build_scenario

    def office_only(name, cfg=None):
        assert name == "office", f"unselected layout was built: {name}"
        return build(name, cfg)

    monkeypatch.setattr(scenarios, "build_scenario", office_only)
    workload = materialize_cases(contract, public.test, tmp_path / "selected")
    assert len(workload["cases"]) == 2
    assert {case["name"] for case in workload["cases"]} == {"office"}


def test_capped_cases_redistribute_exhausted_groups():
    cases = [{"name": "doorway", "map": 0, "pair": 0}] + [
        {"name": "rooms", "map": 1, "pair": pi} for pi in range(40)]
    selected = select_cases(cases, 32, 42)
    assert len(selected) == 32
    assert cases[0] in selected
    assert len({(c["map"], c["pair"]) for c in selected}) == 32


def test_case_selection_and_show_do_not_change_workload(train_cfg, tmp_path):
    materialize(train_cfg, tmp_path / "robot")
    request = compose_config().test
    request.suite = "validation"
    request.max_cases = None
    a = tmp_path / "a"
    a.mkdir()
    first = materialize_cases(train_cfg, request, a)
    request.show, request.sim = True, "both"
    b = tmp_path / "b"
    b.mkdir()
    second = materialize_cases(train_cfg, request, b)
    assert first == second
    request.case = first["cases"][0]["name"]
    c = tmp_path / "c"
    c.mkdir()
    selected = materialize_cases(train_cfg, request, c)
    assert selected["cases"] == [case for case in first["cases"] if case["name"] == request.case]
    # Replay one exact case from frozen maps; selecting by generator alone is not a replay.
    target = first["cases"][-1]
    request.case = None
    request.case_id = target["id"]
    request.dataset = str(a / "maps")
    request.max_cases = 1
    replay = tmp_path / "replay"
    replay.mkdir()
    assert materialize_cases(train_cfg, request, replay)["cases"] == [target]


def test_explicit_reset_uses_materialized_pose(train_cfg, tmp_path):
    from dwarl.training.environment.nav_env import NavVecEnv

    train_cfg.env.num_envs = 1
    env = NavVecEnv(train_cfg, sequential=True)
    try:
        path = env.dataset.maps[0].path(0)
        obs = env.reset_case(0, 0, 1.234, 12)
        first = obs.clone()
        xy, yaw = env.world.robot_pose()
        np.testing.assert_allclose(xy[0], path[0], atol=1e-6)
        np.testing.assert_allclose(yaw[0], 1.234)
        env.reset_case(1, 0, -2., 88)
        repeated = env.reset_case(0, 0, 1.234, 12)
        np.testing.assert_array_equal(first["lidar"], repeated["lidar"])
    finally:
        env.close()


def test_common_scoring_and_infrastructure_exclusion():
    case = {"id": "case", "reference_path": [[0, 0], [3, 4]]}
    success = score(case, success=True, collision=False, timeout=False, travelled=10., elapsed=20., clearance=.1)
    assert success["reference_path_length"] == 5.
    assert success["path_efficiency"] == .5
    collision = score(case, success=True, collision=True, timeout=False, travelled=1., elapsed=2., clearance=0.)
    assert not collision["success"]
    timeout = score(case, success=False, collision=False, timeout=True, travelled=0., elapsed=30., clearance=None)
    assert timeout["clearance"] is None
    rows = [success, collision, timeout, {"case_id": "missing", "status": "infrastructure_failure"}]
    result = summarize(rows, 4)
    assert result["success_rate"] == 1/3
    assert not result["complete"] and result["infrastructure_failures"] == 1
    interrupted = summarize([{"status": "interrupted", "case_id": "case"}], 1)
    assert interrupted["interrupted"] == 1
    assert interrupted["infrastructure_failures"] == 0
    assert interrupted["success_rate"] is None
    assert result["mean_localization_error_m"] is None
    success["diagnostics"] = {"final_pose_comparison": {"localization_difference_m": 0.3}}
    timeout["diagnostics"] = {"final_pose_comparison": {"localization_difference_m": 0.1}}
    assert abs(summarize(rows, 4)["mean_localization_error_m"] - 0.2) < 1e-9
