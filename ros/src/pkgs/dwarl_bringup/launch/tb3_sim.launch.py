"""Robot simulator for the builtin `diffbot` profile: Gazebo + TurtleBot3 waffle (nav2_minimal_tb3_sim).

Reference implementation of the ``ros.sim_launch`` contract used by bench.launch.py: start Gazebo with
``world`` and spawn the robot at (x, y, yaw), publish /clock, odom -> base TF, the scan and accept cmd_vel.

    ros2 launch dwarl_bringup tb3_sim.launch.py world:=<sdf> x:=0 y:=0 yaw:=0 gui:=true
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node


def _setup(context):
    get = lambda k: LC(k).perform(context)  # noqa: E731
    tb3 = get_package_share_directory("nav2_minimal_tb3_sim")
    gz_flags = "-r" if get("gui") == "true" else "-r -s"
    return [
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
            ),
            launch_arguments={"gz_args": f"{gz_flags} {get('world')}", "on_exit_shutdown": "true"}.items(),
        ),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(os.path.join(tb3, "launch", "spawn_tb3.launch.py")),
            launch_arguments={"x_pose": get("x"), "y_pose": get("y"), "yaw": get("yaw")}.items(),
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[
                {"use_sim_time": True, "robot_description": Path(tb3, "urdf", "turtlebot3_waffle.urdf").read_text()}
            ],
        ),
    ]


def generate_launch_description():
    args = [
        ("world", "", "Gazebo world .sdf"),
        ("x", "0.0", ""),
        ("y", "0.0", ""),
        ("yaw", "0.0", ""),
        ("gui", "true", "Gazebo GUI"),
    ]
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in args]
        + [OpaqueFunction(function=_setup)]
    )
