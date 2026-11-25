import json

import numpy as np
import pytest

from dwarl.datasets.generation.mapdata import MapData
from dwarl.testing.results import write_report
from dwarl.testing.scoring import score
from dwarl.testing.trace import BehaviorTrace, path_progress, persist_episode


@pytest.fixture
def case():
    return {"id": "1234567890abcdef", "name": "doorway", "repeat": 0,
            "map_file": "map.npz", "start": [0., 0., 0.], "goal": [2., 0.],
            "reference_path": [[0., 0.], [1., 0.], [2., 0.]]}


def make_trace(case):
    return BehaviorTrace(case, pose_source="test physics", velocity_source="test sensor",
                         command_source="post-safety test command", clearance_source="test geometry")


def test_trace_units_null_values_and_reference_progress(case):
    trace = make_trace(case)
    trace.sample(0, case["start"])
    sample = trace.sample(.1, [1., .2, .3], command=(0., 0.), velocity=(np.nan, np.inf), clearance=0.)
    assert sample["command_v_m_s"] == sample["clearance_m"] == 0.
    assert sample["measured_v_m_s"] is None
    assert sample["measured_w_rad_s"] is None
    assert sample["path_remaining_m"] == 1.
    assert sample["path_completion"] == .5
    assert sample["distance_to_goal_m"] == pytest.approx(np.hypot(1., .2))
    assert trace.data["samples"][0]["command_v_m_s"] is None
    assert trace.data["units"]["yaw_rad"] == "rad"
    json.dumps(trace.data, allow_nan=False)
    assert path_progress([[0, 0], [0, 0]], [0, 0]) == (None, None)


@pytest.mark.parametrize("flags,outcome", [
    ({"success": True, "collision": True, "timeout": False}, "collision"),
    ({"success": False, "collision": False, "timeout": True}, "timeout"),
    ({"success": False, "collision": False, "timeout": False, "stuck": True}, "stuck"),
    ({"success": False, "collision": False, "timeout": False, "termination_reason": "nav2_aborted"}, "aborted"),
    ({"success": False, "collision": False, "timeout": False, "termination_reason": "flipped"}, "terminated"),
])
def test_outcome_is_explicit(case, flags, outcome):
    row = score(case, travelled=1, elapsed=2, clearance=None, **flags)
    assert row["outcome"] == outcome


def test_interruption_preserves_trace_and_missing_outcomes(tmp_path, case):
    trace = make_trace(case)
    trace.sample(0, case["start"])
    trace.sample(.1, [.5, 0., .2], clearance=.2)
    row = persist_episode(tmp_path, "mujoco", case, {"status": "interrupted", "case_id": case["id"],
                          "_trace": trace.data, "completion_time": .1}, "checkpoint-hash")
    assert row["outcome"] == "interrupted"
    assert row["success"] is None and row["collision"] is None
    assert row["safety_stops"] is None
    assert row["final_pose"] == [.5, 0., .2]
    assert row["path_completion"] == .25
    assert row["final_distance_to_goal"] == 1.5
    assert "_trace" not in row
    saved = json.loads((tmp_path / row["trace_file"]).read_text())
    assert saved["checkpoint"] == "checkpoint-hash"
    assert saved["events"][-1]["details"]["outcome"] == "interrupted"
    assert saved["events"][-1]["time_s"] is None  # interruption time is not the last measurement time
    assert saved["events"][-1]["details"]["last_observed_time_s"] == .1


def test_report_separate_images_and_readable_json(tmp_path, case):
    md = MapData(occupancy=np.zeros((8, 8), dtype=bool), resolution=1., size=8., height=1.,
                 primitives=np.array([[0, 2, 2, 0, .2, .5]], dtype=float), generator="doorway", seed=1, density=0.)
    md.save(tmp_path / "map.npz")
    trace = make_trace(case)
    trace.sample(0, case["start"], velocity=(0., 0.), clearance=.5)
    trace.sample(.1, [1., 0., 0.], command=(.2, .3), velocity=(.1, .2), clearance=0.)
    trace.event("safety_stop", .1, [1., 0., 0.])
    row = score(case, success=False, collision=True, timeout=False, travelled=1, elapsed=.1, clearance=0,
                trajectory=[[0, 0], [1, 0]], safety_stops=1)
    row["_trace"] = trace.data
    from dwarl.testing.compute import ComputeProfile

    row["compute"] = ComputeProfile(0, .1).result(.1)
    persist_episode(tmp_path, "gazebo", case, row, "exact-hash")
    failed = persist_episode(tmp_path, "mujoco", case,
                             {"status": "infrastructure_failure", "case_id": case["id"], "error": "no sensor"},
                             "exact-hash")
    write_report(tmp_path, {"cases": [case]}, {"gazebo": [row], "mujoco": [failed]},
                 {"checkpoint": "exact-hash", "workload": "same-cases", "status": "infrastructure_failure"})
    folder = (tmp_path / row["episode_file"]).parent
    assert (folder / "trajectory.png").stat().st_size > 1000
    assert (folder / "motion.png").stat().st_size > 1000
    assert (folder / "compute.png").stat().st_size > 1000
    assert (folder / "compute.json").is_file()
    text = (tmp_path / "report/README.md").read_text()
    assert "post-safety test command" in text
    assert "No trajectory measurements available" in text
    assert "Goal gap (m)" in text
    saved = json.loads((folder / "trace.json").read_text())
    assert [e["type"] for e in saved["events"]] == ["safety_stop", "collision", "termination"]
    assert saved["events"][1]["pose"] == [1., 0., 0.]
