import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "iterate_policy", Path(__file__).parents[1] / "scripts/iterate_policy.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def report(**updates):
    value = {"suite": "validation", "status": "completed", "workload": "frozen-cases",
             "simulators": {"mujoco": {"complete": True, "success_rate": .8, "collision_rate": .1}}}
    value.update(updates)
    return value


def test_policy_selection_rejects_test_data_and_changed_workloads():
    assert MODULE.comparable_score(report(), "frozen-cases")["success_rate"] == .8
    with pytest.raises(ValueError, match="validation"):
        MODULE.comparable_score(report(suite="test"))
    with pytest.raises(ValueError, match="workload changed"):
        MODULE.comparable_score(report(workload="different-cases"), "frozen-cases")
    with pytest.raises(ValueError, match="completed"):
        MODULE.comparable_score(report(status="interrupted"))


def test_policy_selection_requires_collision_evidence():
    value = report()
    value["simulators"]["mujoco"]["collision_rate"] = None
    with pytest.raises(ValueError, match="incomplete"):
        MODULE.comparable_score(value)
