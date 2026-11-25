"""Serializable findings used by the CLI and onboarding."""

from dwarl.cli import ui


def inspect(reference):
    from dwarl.robot.build import build_xml, check_against_urdf, summarize
    from dwarl.robot.profile import load_profile

    report = {"schema": "dwarl_checks/1", "robot": str(reference), "findings": [], "summary": None}
    try:
        profile = load_profile(reference)
        xml = build_xml(profile)
        summary = summarize(profile, xml)
        report.update(robot=profile.id, profile=str(profile.path), fingerprint=profile.sha256,
                      summary=summary.as_dict())
        report["findings"].extend([
            {"level": "pass", "message": "Description and training model compiled"},
            {"level": "pass", "message": f"Drive: radius {summary.wheel_radius:.3f} m · "
             f"separation {summary.wheel_separation:.3f} m"},
            {"level": "pass", "message": f"Mass {summary.mass:.2f} kg · spawn height {summary.spawn_height:.3f} m"},
            {"level": "pass", "message": f"Lidar position {summary.lidar_pos} m · "
             f"upside-down: {summary.lidar_upside_down}"},
        ])
        for warning in check_against_urdf(profile, summary):
            report["findings"].append({"level": "check", "message": warning,
                                       "hint": "Verify these values against the physical robot."})
        report["findings"].append({"level": "check", "message": "Inspect support/caster geometry on the floor",
                                   "hint": ui.command("dwarl", "robot", "view", profile.path)})
    except (ValueError, OSError, KeyError) as exc:
        report["findings"].append({"level": "fail", "message": str(exc),
                                   "hint": "Check the profile and description; rerun robot init to revise answers."})
    report["passed"] = not any(item["level"] == "fail" for item in report["findings"])
    return report


def render(report):
    ui.title("Robot check", f"Robot: {report['robot']}")
    for item in report["findings"]:
        ui.finding(item["level"], item["message"], item.get("hint"))


def environment(sim):
    report = {"schema": "dwarl_checks/1", "sim": sim, "findings": []}
    try:
        if sim == "gazebo":
            from dwarl.testing.backends.gazebo import prerequisites

            report["findings"].extend(prerequisites(quiet=True))
        else:
            import mujoco
            import onnxruntime

            report["findings"].append({"level": "pass", "message":
                                       f"MuJoCo {mujoco.__version__} · ONNX Runtime {onnxruntime.__version__}"})
    except (ValueError, OSError, ImportError) as exc:
        report["findings"].append({"level": "fail", "message": str(exc), "hint": ui.command("setup")})
    report["passed"] = not any(item["level"] == "fail" for item in report["findings"])
    return report
