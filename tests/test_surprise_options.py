import pytest

from dwarl.navigation.options import surprise_obstacles


@pytest.mark.parametrize("release_time", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_release_time_must_be_finite_and_positive(release_time):
    with pytest.raises(ValueError, match="release_time"):
        surprise_obstacles({"probability": 0, "release_time": release_time})


def test_positive_release_time_is_accepted():
    assert surprise_obstacles({"release_time": 0.01})["release_time"] == 0.01
