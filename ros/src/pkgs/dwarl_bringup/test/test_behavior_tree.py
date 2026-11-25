# Copyright (C) 2026 Nobleo Autonomous Solutions B.V.

"""Check that goal-triggered planning retains the collision safety recovery tree."""

from pathlib import Path
from xml.etree import ElementTree


def test_safety_tree_plans_on_goal_updates_without_periodic_replanning():
    tree = ElementTree.parse(
        Path(__file__).parents[1] / "behavior_trees" / "dwarl_safety_retry.xml"
    ).getroot()
    assert tree.find(".//RateController") is None
    planning = tree.find(".//GoalUpdatedController")
    assert planning is not None
    assert planning.find("./RecoveryNode/ComputePathToPose") is not None
    assert tree.find(".//RecoveryNode[@name='SafetyRetry']//FollowPath") is not None
    assert tree.find(".//RecoveryNode[@name='SafetyRetry']//BackUp") is not None
    assert tree.find(".//RecoveryNode[@name='NavigateRecovery']/Sequence") is not None
