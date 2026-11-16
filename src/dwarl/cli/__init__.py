"""Argument adapter; workflows own execution. Help stays light enough for docs and completion."""

import argparse
import json
import os
import sys


def common(parser):
    output = parser.add_argument_group("Output")
    for flag, help_text in (("plain", "plain, non-animated output"), ("json", "structured result on stdout"),
                            ("verbose", "show detailed diagnostics and traceback")):
        output.add_argument(f"--{flag}", action="store_true", default=argparse.SUPPRESS, help=help_text)
    return parser


def parser():
    from rich_argparse import RawDescriptionRichHelpFormatter as RichHelpFormatter

    root = common(argparse.ArgumentParser(
        prog="dwarl", description="Train, test and deploy a local planner for your robot.",
        formatter_class=RichHelpFormatter,
        epilog="Getting started:  setup · robot init\n"
        "Daily workflow:   train · test · deploy\n"
        "Inspection:       status · next · runs · report · doctor\n\n"
        "Start: pixi run setup\nNext step: pixi run dwarl next\nDocs: https://nobleo.github.io/dwarl/"))
    root.set_defaults(plain=False, json=False, verbose=False)
    commands = root.add_subparsers(dest="command", title="Commands", metavar="COMMAND")

    def command(name, help_text, example):
        return common(commands.add_parser(name, help=help_text, description=help_text,
                                           formatter_class=RichHelpFormatter, epilog=f"Example: {example}"))

    p = command("setup", "Get started: try an example, integrate a robot or continue", "pixi run setup")
    p.add_argument("--robot-ws", help="robot repository containing src/")
    p.add_argument("--packages", nargs="+", help="integration packages to build")
    p.add_argument("--example", action="store_true", help="select the included diffbot")
    p.add_argument("--yes", action="store_true", help="use complete, unambiguous saved/detected inputs")
    p.add_argument("--no-build", action="store_true", help="save selection without building ROS")
    for name in ("train", "test", "deploy"):
        p = command(name, {"train": "Train a policy", "test": "Evaluate an exact checkpoint",
                           "deploy": "Install a compatible policy bundle"}[name], f"pixi run {name} --help")
        p.add_argument("--config", help="YAML configuration (replaces ./dwarl.yaml)")
        p.add_argument("--set", action="append", default=[], metavar="PATH=VALUE", help="advanced schema override")
        p.add_argument("--print-config", action="store_true", help="print resolved configuration and exit")
        p.add_argument("--dry-run", action="store_true", help="inspect selected inputs without executing")
        if name in ("train", "deploy"):
            p.add_argument("robot", nargs="?", help="integration/profile (default: setup selection)")
        if name in ("test", "deploy"):
            p.add_argument("--run", help="run directory (default: selected robot's navigation run, then smoke)")
            p.add_argument("--checkpoint", help="latest, best or iteration (default: latest)")
        if name == "train":
            p.add_argument("--preset", choices=("smoke", "navigation"), help="explicit starter recipe")
            for flag in ("num-envs", "seed", "max-iterations"):
                p.add_argument(f"--{flag}", type=int)
            for flag in ("device", "name", "resume"):
                p.add_argument(f"--{flag}")
            p.add_argument("--detached", action="store_true", default=None)
        if name == "test":
            p.add_argument("--sim", choices=("mujoco", "gazebo", "both"))
            p.add_argument("--suite", choices=("benchmark", "validation", "test"))
            p.add_argument("--case")
            p.add_argument("--repeats", type=int)
            p.add_argument("--controllers", type=lambda value: value.split(","), help="dwarl,mppi,dwb,rpp (Gazebo)")
            p.add_argument("--profile-compute", action="store_true", default=None)
        if name in ("train", "test"):
            p.add_argument("--show", action="store_true", default=None)
    for name in ("status", "next"):
        p = command(name, "Inspect progress" if name == "status" else "Show the next useful step",
                    f"pixi run dwarl {name}")
        p.add_argument("--robot", help="inspect an integration without changing the saved selection")
        p.add_argument("--run", help="inspect a particular run")
    p = command("runs", "List, inspect or watch training runs", "pixi run dwarl runs watch")
    p.add_argument("action", choices=("list", "show", "watch"), nargs="?", default="list")
    p.add_argument("run", nargs="?", help="run directory (default: selected run)")
    p.add_argument("--interval", type=float, default=2., help="watch refresh interval in seconds")
    p = command("report", "Summarize or open a navigation report", "pixi run dwarl report --open")
    p.add_argument("directory", nargs="?", help="test directory (default: latest test of selected run)")
    p.add_argument("--run")
    p.add_argument("--open", action="store_true", help="open the local HTML report in a browser")
    p = command("doctor", "Check the environment or a running simulator integration",
                "pixi run dwarl doctor --sim gazebo --live")
    p.add_argument("--sim", choices=("mujoco", "gazebo"), default="mujoco")
    p.add_argument("--robot")
    p.add_argument("--live", action="store_true", help="launch a managed simulator and verify its interfaces")
    p.add_argument("--timeout", type=float, default=120.)
    p = command("robot", "Create or inspect a robot integration", "pixi run dwarl robot init")
    robot = p.add_subparsers(dest="action", required=True, metavar="ACTION")
    for action in ("check", "view"):
        r = common(robot.add_parser(action, help=f"{action.capitalize()} the selected robot",
                                     formatter_class=RichHelpFormatter))
        r.add_argument("robot", nargs="?", help="integration/profile (default: setup selection)")
    r = common(robot.add_parser("init", help="Create or update a robot-owned integration",
                                 formatter_class=RichHelpFormatter))
    r.add_argument("directory", nargs="?")
    r.add_argument("--answers", help="input robot profile YAML; relative assets resolve beside this file")
    r.add_argument("--name", help="ROS package name (default: directory name)")
    r.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    r.add_argument("--yes", action="store_true", help="non-interactive; required inputs must be supplied")
    r.add_argument("--update", action="store_true", help="apply displayed changes to generated files")
    r.add_argument("--profile-only", action="store_true", help="create a profile without ROS package scaffolding")
    r.add_argument("--license", default="Apache-2.0", help="license for the generated robot package")
    r.add_argument("--maintainer")
    r.add_argument("--email")
    r.add_argument("--gazebo", action="store_true", help="wrap an existing robot-only simulator launch")
    r.add_argument("--sim-package")
    r.add_argument("--sim-launch")
    r.add_argument("--arg-map", action="append", default=[], metavar="ARG=SIM_ARG")
    p = command("stop", "Interrupt an owned training process", "pixi run dwarl stop runs/<run>")
    p.add_argument("run")
    p = command("completion", "Print shell-completion setup", "pixi run dwarl completion bash")
    p.add_argument("shell", choices=("bash", "zsh", "fish"), nargs="?", default="bash")
    p = command("model-benchmark", "Replay recorded ONNX inputs", "pixi run dwarl model-benchmark --help")
    p.add_argument("bundle")
    p.add_argument("observations")
    p.add_argument("--output", default="model-benchmark.json")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    return root


def app():
    import contextlib

    import argcomplete
    from omegaconf.errors import OmegaConfBaseException

    from dwarl.cli import ui

    if "--plain" in sys.argv:
        os.environ["NO_COLOR"] = "1"
    command_parser = parser()
    argcomplete.autocomplete(command_parser)
    # Repeated scalar options are usually accidental; repeated --set/--arg-map are intentional.
    seen = set()
    for token in sys.argv[1:]:
        flag = token.partition("=")[0]
        if flag.startswith("--") and flag not in ("--set", "--arg-map"):
            if flag in seen:
                command_parser.error(f"duplicate CLI assignment: {flag}")
            seen.add(flag)
    args = command_parser.parse_args()
    ui.configure(args.plain, args.json, args.verbose)
    if args.command is None:
        if args.json:
            ui.result({"next": "pixi run setup", "help": command_parser.format_help()})
        else:
            command_parser.print_help()
        return
    code, payload = 0, None
    try:
        data_output = args.command == "completion" or getattr(args, "print_config", False)
        with contextlib.redirect_stdout(sys.stderr) if args.json or not data_output else contextlib.nullcontext():
            code, payload = dispatch(args)
    except (ValueError, OSError, KeyError, OmegaConfBaseException) as exc:
        if args.verbose:
            ui.console.print_exception(show_locals=False)
        else:
            ui.finding("fail", str(exc))
        code, payload = 2, {"schema": "dwarl_error/1", "error": str(exc)}
    except (KeyboardInterrupt, EOFError):
        ui.info("Interrupted. Completed checkpoints, available results and wizard drafts are retained.")
        code, payload = 130, {"status": "interrupted"}
    except Exception as exc:
        import time
        import traceback
        from pathlib import Path

        log = Path(".dwarl/logs") / f"error-{time.time_ns()}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(traceback.format_exc())
        ui.finding("fail", str(exc), f"Details: {log}. Rerun with --verbose for a traceback.")
        if args.verbose:
            ui.console.print_exception(show_locals=False)
        code, payload = 2, {"schema": "dwarl_error/1", "error": str(exc), "log": str(log)}
    if payload is not None:
        ui.result(payload)
    raise SystemExit(code or 0)


def selected_robot(explicit=None):
    from dwarl.workflows.context import load

    reference = explicit or load().get("robot")
    if not reference:
        raise ValueError("No robot selected. Start with `pixi run setup` or pass an integration path.")
    return reference


def dispatch(args):
    from omegaconf import OmegaConf

    from dwarl.cli import ui
    from dwarl.config import compose_config
    from dwarl.workflows import context

    if args.command not in ("train", "test", "deploy"):
        return utility(args)
    assignments = list(args.set)
    excluded = {"command", "config", "set", "print_config", "dry_run", "json", "plain", "verbose", "preset"}
    for name, value in vars(args).items():
        if name in excluded or value is None:
            continue
        key = "robot" if name == "robot" else f"{args.command}.{name}"
        assignments.append(f"{key}={json.dumps(value)}")
    keys = []
    preset = getattr(args, "preset", None)
    cfg = compose_config(args.config, assignments, explicit_keys=keys, preset=preset)
    if preset and cfg.train.resume:
        raise ValueError("Resume restores the saved recipe; omit --preset when using --resume.")
    if cfg.robot is None and "robot" not in keys:
        cfg.robot = context.load().get("robot")
    if args.command in ("test", "deploy") and f"{args.command}.run" not in keys:
        run = context.selected_run(cfg.robot)
        if run:
            cfg[args.command].run = run
        elif cfg.robot and not args.print_config:
            raise ValueError("No run recorded for this robot. Train it first or select a run with --run.")
    if args.print_config:
        if args.json:
            return 0, OmegaConf.to_container(cfg, resolve=True)
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return 0, None
    if args.dry_run:
        from dwarl.workflows.preflight import execute

        execute(args.command, cfg)
        return 0, {"command": args.command, "config": OmegaConf.to_container(cfg, resolve=True)}
    ui.title(args.command.capitalize(), f"Robot: {cfg.robot or 'frozen run profile'}")
    if args.command == "train":
        if cfg.train.detached:
            from dwarl.runs.process import detach

            return 0, detach(cfg)
        from dwarl.workflows.train import execute

        run = execute(cfg, keys)
        state = json.loads((run / "status.json").read_text())
        return 130 if state["state"] == "interrupted" else 0, {"run": str(run), **state}
    if args.command == "test":
        if cfg.test.sim in ("gazebo", "both"):
            from dwarl.runs.store import resolve_run
            from dwarl.workflows.prepare import ros

            frozen = resolve_run(cfg.test.run) / "robot/source_profile.yaml"
            # Source profile path is recorded in provenance, since installed launch packages are live.
            provenance = json.loads((frozen.parents[1] / "provenance.json").read_text())
            ros(provenance.get("source_profile"))
        from dwarl.workflows.test import execute

        out, report = execute(cfg)
        code = 130 if report["status"] == "interrupted" else (
            (0 if report["passed"] else 1) if report["status"] == "completed" else 2)
        return code, {"directory": str(out), **report}
    from dwarl.workflows.deploy import execute

    target = execute(cfg)
    return 0, {"bundle": str(target)}


def utility(args):
    from dwarl.cli import ui

    if args.command == "setup":
        from dwarl.workflows.setup import execute

        code = execute(args.robot_ws, args.packages, interactive=False if args.yes else None,
                       build=not args.no_build, example=args.example)
        return code, {"completed": code == 0}
    if args.command in ("status", "next", "runs"):
        from dwarl.workflows.guidance import execute

        return execute(args)
    if args.command == "report":
        from dwarl.workflows.report import execute

        return 0, execute(args.directory, args.run, args.open)
    if args.command == "doctor":
        from dwarl.robot.diagnostics import environment

        if args.live and args.sim != "gazebo":
            raise ValueError("--live requires --sim gazebo.")
        if args.sim == "gazebo":
            from dwarl.workflows.prepare import ros

            ros(selected_robot(args.robot) if args.live or args.robot else None)
        if args.live:
            from dwarl.workflows.sim_check import execute

            report = execute(selected_robot(args.robot), args.timeout)
        else:
            report = environment(args.sim)
        for item in report["findings"]:
            ui.finding(item["level"], item["message"], item.get("hint"))
        return 0 if report["passed"] else 2, report
    if args.command == "robot":
        if args.action == "init":
            from dwarl.robot.init import execute

            report = execute(args.directory, answers=args.answers, name=args.name, assignments=args.set,
                             interactive=sys.stdin.isatty() and not args.yes, update=args.update,
                             gazebo=args.gazebo, sim_package=args.sim_package, sim_launch=args.sim_launch,
                             arg_map=args.arg_map, license_id=args.license, maintainer=args.maintainer,
                             email=args.email, profile_only=args.profile_only)
            return 0, report
        reference = selected_robot(args.robot)
        from dwarl.robot.diagnostics import inspect, render

        report = inspect(reference)
        render(report)
        if report["passed"] and args.action == "view":
            import mujoco
            import mujoco.viewer

            from dwarl.robot.build import build_xml
            from dwarl.robot.profile import load_profile
            from dwarl.robot.view import inspection_model

            model, data = inspection_model(build_xml(load_profile(reference)))
            ui.info("Inspect simplified training geometry on a floor. Close the viewer to continue.")
            mujoco.viewer.launch(model, data)
        elif report["passed"]:
            ui.next_step("dwarl", "robot", "view", reference)
        return 0 if report["passed"] else 2, report
    if args.command == "completion":
        import subprocess

        output = subprocess.check_output(["register-python-argcomplete", "--shell", args.shell, "dwarl"], text=True)
        print(output)
        ui.info("Evaluate this in your pixi shell, or add it to your shell configuration.")
        return 0, {"shell": args.shell, "script": output}
    if args.command == "stop":
        from dwarl.runs.process import stop

        stop(args.run)
        ui.info("Interrupt requested; completed checkpoints are retained.")
        return 0, {"run": args.run, "status": "interrupt_requested"}
    from dwarl.testing.model_benchmark import execute

    execute(args.bundle, args.observations, args.output, args.threads, args.iterations, args.warmup)
    return 0, {"output": args.output}
