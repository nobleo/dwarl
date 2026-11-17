from types import SimpleNamespace

import numpy as np
import pytest
from dwarl_runtime import Scan, lidar_observation


@pytest.mark.parametrize("offset", [-1e-13, 0.0, 1e-13])
def test_partial_scan_preserves_first_ray_at_roundoff_boundary(offset):
    spec = SimpleNamespace(num_rays=3, angle_min=-1.0, angle_increment=0.5,
                           origin=(0.12, 0.0), full_circle=False, min_range=0.0,
                           max_range=10.0, lidar_normalize="linear")
    scan = Scan(np.array([2.0, 4.0, 6.0]), spec.angle_min + offset, 0.5, 0.0, 10.0)
    actual = lidar_observation(spec, scan, (0.12, 0.0, 0.0))
    np.testing.assert_allclose(actual, [0.2, 0.4, 0.6], atol=1e-7)


def test_partial_scan_still_excludes_returns_outside_field_of_view():
    spec = SimpleNamespace(num_rays=3, angle_min=-1.0, angle_increment=0.5,
                           origin=(0.12, 0.0), full_circle=False, min_range=0.0,
                           max_range=10.0, lidar_normalize="linear")
    scan = Scan(np.array([2.0]), spec.angle_min - 1e-6, 0.5, 0.0, 10.0)
    np.testing.assert_array_equal(lidar_observation(spec, scan, (0.12, 0.0, 0.0)), np.ones(3))
