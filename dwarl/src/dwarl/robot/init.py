"""Resumable robot-owned integration generator; physical measurements are supplied by the user."""

import difflib
import hashlib
import json
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from importlib.resources import files
from pathlib import Path
from xml.sax.saxutils import escape

import yaml
from omegaconf import OmegaConf

from dwarl.cli import ui
from dwarl.robot.profile import PROFILE_FILE, RobotProfile
from dwarl.runs.store import write_json


def set_value(data, key, value):
    parts = key.split(".")
    for part in parts[:-1]:
        data = data.setdefault(part, {})
    data[parts[-1]] = value


def get_value(data, key, default=None):
    for part in key.split("."):
        if not isinstance(data, dict) or part not in data:
            return default
        data = data[part]
    return data


def positive(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Enter a positive finite number.")
    return value


def finite(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("Enter a finite number.")
    return value


def mapping(text):
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Enter a YAML mapping, for example {use_sim: true}.")
    return value


def boolean(text):
    if text.lower() not in ("true", "false"):
        raise ValueError("Choose true or false.")
    return text.lower() == "true"


def questionnaire(data, fields, draft):
    """Fields: key, label, default, parser, completion choices. :back revisits a field."""
    index = 0
    while index < len(fields):
        key, label, default, parse, choices = fields[index]
        value = get_value(data, key, default)
        if isinstance(value, (dict, list, bool)):
            value = json.dumps(value)
        try:
            answer = ui.ask(label, "" if value is None else value, choices=choices, path=key == "description.urdf")
            if not answer:
                raise ValueError("This value is required; use your robot's actual measurement.")
            if choices and answer not in choices:
                raise ValueError(f"Choose one of: {', '.join(choices)}")
            set_value(data, key, parse(answer))
            write_json(draft, data)
            index += 1
        except ui.Back:
            index = max(0, index - 1)
        except (ValueError, TypeError, yaml.YAMLError) as exc:
            ui.finding("check", str(exc))


def interactive_profile(data, target, draft):
    from dwarl.robot.build import expand_urdf

    ui.info("Press Tab for choices; :back revisits a field. Ctrl+C saves your draft.")
    ui.title("Description")
    workspace = next((p for p in target.parents if p.name == "src"), target.parent)
    descriptions = sorted(p for p in workspace.rglob("*") if p.is_file()
                          and (p.name.endswith(".urdf") or p.name.endswith(".urdf.xacro"))
                          and not {"build", "install", ".pixi", ".git"}.intersection(p.relative_to(workspace).parts))
    if descriptions and not get_value(data, "description.urdf"):
        ui.info("Description candidates:\n" + "\n".join(f"  {p}" for p in descriptions[:12]))
    default_description = str(descriptions[0]) if len(descriptions) == 1 else None
    questionnaire(data, [
        ("id", "Robot ID", target.name.removesuffix("_dwarl"), str, ()),
        ("description.urdf", "URDF/xacro path or package:// URI", default_description, str, ()),
        ("description.xacro_args", "Xacro arguments (YAML mapping)", {}, mapping, ()),
    ], draft)
    uri = data["description"]["urdf"]
    if not uri.startswith(("package://", "$(find ")):
        data["description"]["urdf"] = str(Path(uri).expanduser().resolve())
    root = ET.fromstring(expand_urdf(RobotProfile(target / PROFILE_FILE, data)))
    links = [el.attrib["name"] for el in root.findall("link")]
    joints = [el.attrib["name"] for el in root.findall("joint") if el.get("type") in ("continuous", "revolute")]

    def candidate(values, tokens):
        matches = [v for v in values if any(t in v.lower() for t in tokens)]
        return matches[0] if len(matches) == 1 else None

    ui.title("Drive and limits", "Joint/link suggestions come from the description; limits come from your robot.")
    questionnaire(data, [
        ("description.base_link", "Base link", "base_link" if "base_link" in links else None, str, links),
        ("drive.left_wheel_joint", "Left wheel joint", candidate(joints, ("left",)), str, joints),
        ("drive.right_wheel_joint", "Right wheel joint", candidate(joints, ("right",)), str, joints),
        ("limits.v_min", "Minimum forward velocity (m/s; negative enables reversing)", None, finite, ()),
        ("limits.v_max", "Maximum forward velocity (m/s)", None, positive, ()),
        ("limits.w_max", "Maximum turn rate (rad/s)", None, positive, ()),
        ("limits.a_max", "Linear acceleration limit (m/s²)", None, positive, ()),
        ("limits.alpha_max", "Angular acceleration limit (rad/s²)", None, positive, ()),
    ], draft)
    ui.title("Physical footprint", "Dimensions are relative to the base origin; use the actual outer footprint.")
    if "footprint" not in data:
        shape = ui.choose("Footprint input", ["rectangle", "radius", "polygon"])
        if shape == "rectangle":
            size_draft = draft.with_suffix(".footprint.json")
            sizes = json.loads(size_draft.read_text()) if size_draft.exists() else {}
            questionnaire(sizes, [(key, label, None, parse, ()) for key, label, parse in [
                ("length", "Length (m)", positive), ("width", "Width (m)", positive),
                ("x", "Footprint centre x offset (m)", finite), ("y", "Footprint centre y offset (m)", finite),
            ]], size_draft)
            x, y, a, b = sizes["x"], sizes["y"], sizes["length"] / 2, sizes["width"] / 2
            data["footprint"] = {"polygon": [[x+a, y+b], [x+a, y-b], [x-a, y-b], [x-a, y+b]]}
        elif shape == "radius":
            questionnaire(data, [("footprint.radius", "Radius about base origin (m)", None, positive, ())], draft)
        else:
            questionnaire(data, [("footprint.polygon", "Ordered convex polygon [[x,y], ...] (m)",
                                  None, yaml.safe_load, ())], draft)
    else:
        questionnaire(data, [("footprint", "Footprint (YAML mapping)", data["footprint"], mapping, ())], draft)
    ui.title("Sensing", "Use the scan after filtering; URDF supplies the sensor mounting transform.")
    questionnaire(data, [
        ("lidar.link", "Lidar link", candidate(links, ("laser", "lidar")), str, links),
        ("lidar.num_rays", "Training ray count", None, int, ()),
        ("lidar.fov_deg", "Useful field of view (degrees)", None, positive, ()),
        ("lidar.min_range", "Minimum useful range (m)", None, finite, ()),
        ("lidar.max_range", "Maximum useful range (m)", None, positive, ()),
    ], draft)
    ui.title("ROS interfaces", "Verify suggested topics and frames against your robot bringup.")
    questionnaire(data, [(f"ros.{key}", label, default, parse, choices) for key, label, default, parse, choices in [
        ("base_frame", "Base frame", data["description"]["base_link"], str, ()),
        ("odom_frame", "Odometry frame", "odom", str, ()),
        ("map_frame", "Global frame (standard Nav2 uses map)", "map", str, ()),
        ("scan_topic", "Filtered scan topic", "/scan", str, ()),
        ("odom_topic", "Odometry topic", "/odom", str, ()),
        ("cmd_vel_topic", "Command topic", "/cmd_vel", str, ()),
        ("cmd_vel_stamped", "Does the command topic use TwistStamped?", True, boolean, ("true", "false")),
    ]], draft)
    ui.info("Simulation assumptions: actuator kv=1, max_wheel_speed=30 rad/s, max_torque=5 N·m; "
            "lidar noise_std=0.01, dropout_prob=0.005 unless specified. Review them for your model.")
    if ui.ask("Edit actuator/support/physics settings now? (y/n)", "n").lower() == "y":
        questionnaire(data, [
            ("drive.actuator", "Actuator settings", {"kv": 1., "max_wheel_speed": 30., "max_torque": 5.}, mapping, ()),
            ("collision", "Collision/support overrides", {}, mapping, ()),
            ("physics", "Physics overrides", {}, mapping, ()),
        ], draft)
    write_json(draft, data)
    return data


def simulation_wrapper(package, filename, arguments):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", package) or Path(filename).name != filename:
        raise ValueError("Simulator needs a ROS package name and a launch filename (no directories).")
    required = {"world", "x", "y", "yaw", "gui"}
    if set(arguments) != required or len(set(arguments.values())) != 5:
        raise ValueError("Map world, x, y, yaw, gui to five distinct simulator argument names.")
    pairs = ",\n            ".join(f"{destination!r}: LaunchConfiguration({source!r})"
                                     for source, destination in arguments.items())
    return f'''"""Robot-only simulator wrapper. Verify with dwarl doctor --sim gazebo --live."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    source = Path(get_package_share_directory({package!r})) / "launch" / {filename!r}
    defaults = {{"world": "", "x": "0", "y": "0", "yaw": "0", "gui": "false"}}
    return LaunchDescription([
        *(DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(source)), launch_arguments={{
            {pairs}
        }}.items()),
    ])
'''


def required_inputs(data):
    required = ["id", "limits.v_min", "limits.v_max", "limits.w_max", "limits.a_max", "limits.alpha_max",
                "lidar.num_rays", "lidar.fov_deg", "lidar.min_range", "lidar.max_range"]
    if not get_value(data, "description.mjcf"):
        required += ["description.urdf", "drive.left_wheel_joint", "drive.right_wheel_joint", "lidar.link"]
    missing = [key for key in required if get_value(data, key) is None]
    if not get_value(data, "footprint"):
        missing.append("footprint (polygon or radius)")
    if missing:
        raise ValueError("Missing required inputs:\n  " + "\n  ".join(missing))


def execute(directory=None, *, answers=None, name=None, assignments=(), interactive=True, update=False,
            gazebo=False, sim_package=None, sim_launch=None, arg_map=(), license_id="Apache-2.0",
            maintainer=None, email=None, profile_only=False):
    if directory is None:
        directory = ui.ask("Integration directory in your robot repository", path=True, interactive=interactive)
    target = Path(directory).expanduser().resolve()
    package_name = name or target.name
    if not profile_only and not re.fullmatch(r"[a-z][a-z0-9_]*", package_name):
        raise ValueError("Package name must start with a lowercase letter and contain only a-z, 0-9 and underscores.")
    draft = Path(".dwarl/drafts") / (hashlib.sha256(str(target).encode()).hexdigest()[:16] + ".json")
    source = Path(answers).expanduser().resolve() if answers else target / PROFILE_FILE
    data = yaml.safe_load(source.read_text()) if source.exists() else {}
    if not answers and draft.exists():
        data = json.loads(draft.read_text())
        ui.info(f"Continuing saved draft: {draft}")
    if not isinstance(data, dict):
        raise ValueError("Profile input must be a YAML mapping.")
    data.setdefault("schema", "dwarl_robot/1")
    if source.exists():
        for key in ("urdf", "mjcf"):
            uri = get_value(data, f"description.{key}")
            if uri and not uri.startswith(("package://", "$(find ")):
                data["description"][key] = str((source.parent / Path(uri).expanduser()).resolve())
    if assignments:
        data = OmegaConf.to_container(OmegaConf.merge(data, OmegaConf.from_dotlist(list(assignments))), resolve=True)
    if interactive and not (gazebo and source.exists()):
        interactive_profile(data, target, draft)
    required_inputs(data)
    if interactive and not gazebo and not get_value(data, "ros.sim_launch"):
        gazebo = ui.ask("Add an existing Gazebo simulator before training? (y/n)", "n").lower() == "y"
    generated = {}
    if gazebo:
        from dwarl.robot.profile import _package_name

        workspace = next((p for p in target.parents if p.name == "src"), target.parent)
        launches = {}
        for manifest in workspace.rglob("package.xml"):
            if {"build", "install", ".pixi", ".git"}.intersection(manifest.relative_to(workspace).parts):
                continue
            names = [p.name for p in (manifest.parent / "launch").glob("*.launch.py")]
            detected_name = _package_name(manifest)
            if names and detected_name:
                launches[detected_name] = names
        if interactive and launches:
            ui.info("Detected simulator launch packages: " + ", ".join(sorted(k for k in launches if k)))
        sim_package = sim_package or ui.ask("Robot-only simulator package", choices=list(launches),
                                            interactive=interactive)
        sim_launch = sim_launch or ui.ask("Simulator launch filename", choices=launches.get(sim_package, ()),
                                          interactive=interactive)
        if any("=" not in item for item in arg_map):
            raise ValueError("Use --arg-map world=world_file (and x, y, yaw, gui as needed).")
        arguments = dict(item.split("=", 1) for item in arg_map)
        for key in ("world", "x", "y", "yaw", "gui"):
            if key not in arguments:
                arguments[key] = ui.ask(f"Simulator argument for {key}", key, interactive=interactive)
        generated["launch/simulation.launch.py"] = simulation_wrapper(sim_package, sim_launch, arguments)
        set_value(data, "ros.sim_launch", {"package": package_name, "file": "simulation.launch.py"})
        ui.info("The wrapper maps arguments. Run the live doctor to verify behavior before training.")
    if not profile_only:
        set_value(data, "deployment.launch", {"package": package_name, "file": "navigation.launch.py"})
        generated["launch/navigation.launch.py"] = files("dwarl").joinpath("templates/navigation.launch.py").read_text()
        if not (target / "package.xml").exists():
            maintainer = maintainer or ui.ask("Package maintainer", interactive=interactive)
            email = email or ui.ask("Maintainer email", interactive=interactive)
            if "@" not in email:
                raise ValueError("Supply a maintainer email address.")
            dependencies = {"launch", "launch_ros", "ament_index_python", "python3-yaml", "dwarl_bringup"}
            for uri in data.get("description", {}).values():
                if isinstance(uri, str) and uri.startswith("package://"):
                    dependencies.add(uri.removeprefix("package://").split("/")[0])
            if sim_package:
                dependencies.add(sim_package)
            deps = "\n".join(f"  <exec_depend>{escape(dep)}</exec_depend>" for dep in sorted(dependencies))
            generated["package.xml"] = (
                f'<?xml version="1.0"?>\n<package format="3">\n  <name>{package_name}</name>\n'
                f'  <version>0.1.0</version>\n  <description>DWARL robot integration</description>\n'
                f'  <maintainer email="{escape(email, {chr(34): "&quot;"})}">{escape(maintainer)}</maintainer>\n'
                f'  <license>{escape(license_id)}</license>\n  <buildtool_depend>ament_cmake</buildtool_depend>\n'
                f'{deps}\n  <export><build_type>ament_cmake</build_type></export>\n</package>\n')
            generated["CMakeLists.txt"] = (
                f"cmake_minimum_required(VERSION 3.16)\nproject({package_name})\nfind_package(ament_cmake REQUIRED)\n"
                "install(FILES dwarl_robot.yaml DESTINATION share/${PROJECT_NAME})\n"
                "install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})\nament_package()\n")
        else:
            from dwarl.robot.profile import _package_name

            if _package_name(target / "package.xml") != package_name:
                raise ValueError("--name must match the existing package.xml name.")
            ui.info("Existing ROS package: ensure it installs dwarl_robot.yaml and launch/, and declares adapter deps.")
    # Validate in a sibling staging directory so source package discovery retains workspace hints.
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dwarl-check-", dir=target.parent) as staged:
        profile_path = Path(staged) / PROFILE_FILE
        profile_path.write_text(yaml.safe_dump(data, sort_keys=False))
        from dwarl.robot.diagnostics import inspect, render

        checked = inspect(profile_path)
        render(checked)
        if not checked["passed"]:
            write_json(draft, data)
            raise ValueError(f"Model validation failed; draft saved at {draft}. Revise answers and rerun robot init.")
    generated[PROFILE_FILE] = yaml.safe_dump(data, sort_keys=False)
    manifest_path = target / ".dwarl-generated.json"
    owned = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    changes = []
    for relative, content in generated.items():
        destination = target / relative
        if destination.exists() and destination.read_text() != content:
            old = destination.read_text()
            ui.info("".join(difflib.unified_diff(old.splitlines(True), content.splitlines(True),
                                               fromfile=str(destination), tofile="generated")))
            if relative != PROFILE_FILE and owned.get(relative) != hashlib.sha256(old.encode()).hexdigest():
                raise ValueError(f"Preserving user-edited file {destination}; merge the shown changes manually.")
            changes.append(relative)
    if changes and not update:
        if not interactive or ui.ask("Apply these changes? (y/n)", "n").lower() != "y":
            raise ValueError("No integration files changed. Use --update to apply the displayed changes.")
    target.mkdir(parents=True, exist_ok=True)
    for relative, content in generated.items():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".dwarl-tmp")
        temporary.write_text(content)
        os.replace(temporary, destination)
        owned[relative] = hashlib.sha256(content.encode()).hexdigest()
    write_json(manifest_path, owned)
    draft.unlink(missing_ok=True)
    draft.with_suffix(".footprint.json").unlink(missing_ok=True)
    from dwarl.workflows.context import select_robot

    select_robot(target)
    ui.finding("pass", f"Integration ready: {target}")
    if changes:
        ui.info("Existing runs keep their frozen contract; changed robot/ROS settings require a new training run.")
    ui.next_step("dwarl", "robot", "view", target)
    return {"schema": "dwarl_init/1", "profile": str(target / PROFILE_FILE), "files": list(generated)}
