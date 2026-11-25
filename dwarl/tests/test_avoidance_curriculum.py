import numpy as np

from dwarl.navigation.avoidance_curriculum import AvoidanceCurriculum
from dwarl.navigation.options import surprise_obstacles


def report(c, obstacle=False, success=True, collision=False, count=200, stage=None):
    c.report(np.full(count, c.stage if stage is None else stage), np.full(count, obstacle),
             np.full(count, success), np.full(count, collision))


def test_clean_success_unlocks_obstacles_but_clean_only_cannot_advance_further():
    c = AvoidanceCurriculum()
    base = surprise_obstacles({"probability": .6})
    assert c.settings(base)["probability"] == 0
    report(c, count=199)
    assert c.stage == 0
    report(c, count=1)
    assert c.stage == 1
    assert c.settings(base)["probability"] == .12
    assert c.settings(base)["count"] == [1, 1]
    report(c)
    assert c.stage == 1
    report(c, obstacle=True)
    assert c.stage == 2


def test_collision_gate_demotion_and_stale_episodes():
    c = AvoidanceCurriculum()
    report(c, collision=True)
    assert c.stage == 0
    report(c)
    report(c)
    report(c, obstacle=True, stage=0)
    assert c.stage == 1
    report(c, obstacle=True, collision=True)
    assert c.stage == 0


def test_final_mix_and_checkpoint_preserve_partial_windows():
    c = AvoidanceCurriculum()
    report(c)
    report(c, obstacle=True, count=100)
    restored = AvoidanceCurriculum()
    restored.restore(c.state())
    report(restored)
    report(restored, obstacle=True, count=100)
    assert restored.stage == 2
    for _ in range(5):
        report(restored)
        report(restored, obstacle=True)
    assert restored.stage == 5
    base = surprise_obstacles({"probability": .6})
    final = restored.settings(base)
    for key in ("probability", "count", "size", "margin"):
        assert final[key] == base[key]
