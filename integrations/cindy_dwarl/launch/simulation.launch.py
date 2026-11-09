"""Cindy in Gazebo Harmonic: the DWARL ``ros.sim_launch`` adapter.

    ros2 launch cindy_dwarl simulation.launch.py world:=<sdf> x:=0 y:=0 yaw:=0 gui:=true

Starts only the simulator and robot interfaces: Gazebo with ``world``, the robot spawned as entity
``robot`` at (x, y, yaw), robot_state_publisher, ros2_control (diff drive with odom -> base_footprint TF,
odometry starting at zero at spawn), and /clock + /scan bridges. DWARL starts the map, localisation
transform and Nav2.
"""

import os
from pathlib import Path

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node


def _setup(context):
    get = lambda k: LC(k).perform(context)  # noqa: E731
    share = get_package_share_directory("cindy_dwarl")
    description = xacro.process_file(os.path.join(share, "urdf", "cindy_gz.urdf.xacro")).toxml()
    gz_flags = "-r" if get("gui") == "true" else "-r -s --headless-rendering"
    resources = [str(Path(get_package_share_directory(p)).parent) for p in ("cindy_description", "cindy_dwarl")]
    resources += [os.environ["GZ_SIM_RESOURCE_PATH"]] if os.environ.get("GZ_SIM_RESOURCE_PATH") else []
    sim_time = {"use_sim_time": True}
    spawn = Node(package="ros_gz_sim", executable="create", output="screen",
                 arguments=["-name", "robot", "-string", description, "-x", get("x"), "-y", get("y"),
                            "-z", "0.01", "-Y", get("yaw")])
    controllers = [Node(package="controller_manager", executable="spawner", output="screen", parameters=[sim_time],
                        arguments=[name, "--controller-manager-timeout", "60"])
                   for name in ("joint_state_broadcaster", "diff_drive_base_controller")]
    return [
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.pathsep.join(resources)),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(os.path.join(get_package_share_directory("ros_gz_sim"), "launch",
                                                    "gz_sim.launch.py")),
            launch_arguments={"gz_args": f"{gz_flags} {get('world')}", "on_exit_shutdown": "true"}.items()),
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
             parameters=[sim_time, {"robot_description": description}]),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen", parameters=[sim_time],
             arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                        "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"]),
        spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=controllers)),
    ]


def generate_launch_description():
    args = [("world", "", "Gazebo world .sdf"), ("x", "0.0", ""), ("y", "0.0", ""), ("yaw", "0.0", ""),
            ("gui", "true", "Gazebo GUI")]
    return LaunchDescription([DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in args]
                             + [OpaqueFunction(function=_setup)])
