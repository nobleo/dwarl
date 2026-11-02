"""DWARL Gazebo benchmark for any robot: robot simulator + perfect localisation + map + Nav2/dwarl.

    ros2 launch dwarl_bringup bench.launch.py scenario:=u_trap model_dir:=<bundle>      (or run:=latest)
    ros2 launch dwarl_bringup bench.launch.py world:=<sdf> map:=<yaml> x:=1 y:=2 yaw:=0 model_dir:=<bundle>

The robot comes from the bundle: its profile's ``ros.sim_launch`` ({package, file}) is included with
``world:=<sdf> x:= y:= yaw:= gui:=`` and must start Gazebo with the robot spawned there, publishing /clock,
odom -> base TF, the scan and accepting cmd_vel (see dwarl_bringup/launch/tb3_sim.launch.py, the reference
implementation for the builtin `diffbot` profile).
Scenario worlds/maps come from `dwarl scenarios build` ($DWARL_ROOT/data/maps/scenarios/gazebo/<name>.*).
Localisation is perfect: a static map->odom transform at the spawn pose (Gazebo odometry starts at zero).
"""

import json
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from dwarl_bringup.nav import load_deploy, resolve_bundle
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node


def _setup(context):
    get = lambda k: LC(k).perform(context)  # noqa: E731
    bundle = resolve_bundle(get("model_dir"))
    deploy = load_deploy(bundle)
    ros = deploy.get("ros") or {}
    x, y, yaw = float(get("x")), float(get("y")), float(get("yaw"))
    world, map_yaml = get("world"), get("map")
    if not world or not map_yaml:
        raise RuntimeError("provide explicit world:=<sdf> map:=<yaml> x:= y:= yaw:=")
    sim = ros.get("sim_launch") or {}
    pkg, file = get("robot_sim_package") or sim.get("package"), get("robot_sim_file") or sim.get("file")
    if not pkg or not file:
        raise RuntimeError(
            f"robot '{deploy['robot']['id']}' has no ros.sim_launch in its profile; "
            "pass robot_sim_package:= robot_sim_file:="
        )
    here = get_package_share_directory("dwarl_bringup")
    st = {"use_sim_time": True}
    return [
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(os.path.join(get_package_share_directory(pkg), "launch", file)),
            launch_arguments={
                "world": str(world),
                "x": str(x),
                "y": str(y),
                "yaw": str(yaw),
                "gui": get("gui"),
            }.items(),
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            parameters=[st],
            arguments=[
                "--x",
                str(x),
                "--y",
                str(y),
                "--yaw",
                str(yaw),
                "--frame-id",
                ros.get("map_frame", "map"),
                "--child-frame-id",
                ros.get("odom_frame", "odom"),
            ],
        ),
        Node(package="nav2_map_server", executable="map_server", parameters=[st, {"yaml_filename": str(map_yaml)}]),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_map",
            parameters=[st, {"autostart": True, "node_names": ["map_server"]}],
        ),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(os.path.join(here, "launch", "navigation.launch.py")),
            launch_arguments={
                "use_sim_time": "true",
                "model_dir": str(bundle),
                "safety": get("safety"),
                "planner": get("planner"),
                "controller": get("controller"),
                "profile_compute": get("profile_compute"),
                "controller_params": get("controller_params"),
                "params_output": get("params_output"),
            }.items(),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            condition=IfCondition(get("rviz")),
            arguments=["-d", os.path.join(here, "config", "dwarl.rviz")],
            parameters=[st],
        ),
    ]


def generate_launch_description():
    args = [
        ("world", "", "Gazebo world .sdf (without scenario)"),
        ("map", "", "map .yaml (without scenario)"),
        ("x", "0.0", "spawn x (without scenario)"),
        ("y", "0.0", ""),
        ("yaw", "0.0", ""),
        ("model_dir", "", "policy bundle dir; overrides run"),
        ("robot_sim_package", "", "override the profile's ros.sim_launch package"),
        ("robot_sim_file", "", "override the profile's ros.sim_launch file"),
        ("planner", "navfn", "global planner: navfn | smac2d"),
        ("controller", "rl", "local controller: dwarl | mppi | dwb"),
        ("profile_compute", "false", "publish wall-clock controller call timings"),
        ("controller_params", "", "optional controller tuning YAML"),
        ("params_output", "", "optional rendered parameter snapshot path"),
        ("safety", "true", "collision_monitor stop + back-up-and-retry behavior tree"),
        ("gui", "true", "Gazebo GUI"),
        ("rviz", "true", "RViz"),
    ]
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in args]
        + [OpaqueFunction(function=_setup)]
    )
