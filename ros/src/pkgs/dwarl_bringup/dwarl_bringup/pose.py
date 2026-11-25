"""Rigid transforms shared by Gazebo localization and independent footprint scoring."""

import math

import numpy as np
from scipy.spatial.transform import Rotation


def matrix(position, orientation):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(
        [orientation.x, orientation.y, orientation.z, orientation.w]
    ).as_matrix()
    result[:3, 3] = [position.x, position.y, position.z]
    return result


def base_pose(position, orientation, offset):
    world_base = matrix(position, orientation) @ matrix(offset.translation, offset.rotation)
    return [float(world_base[0, 3]), float(world_base[1, 3]), math.atan2(world_base[1, 0], world_base[0, 0])]
