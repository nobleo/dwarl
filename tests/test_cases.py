
import numpy as np

from dwarl.config import compose_config
from dwarl.robot.config import materialize
from dwarl.testing.cases import materialize as materialize_cases
from dwarl.testing.cases import select_cases
from dwarl.testing.scoring import score, summarize


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
