import time
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
from uuid import uuid4

from omegaconf import OmegaConf

from dwarl.artifacts.prepare import prepare_policy
from dwarl.runs.store import resolve_checkpoint, resolve_run, sha256, write_json
from dwarl.testing.cases import materialize
from dwarl.testing.scoring import summarize
from dwarl.testing.trace import persist_episode
from dwarl.training.contract import load_contract


def _variants(request):
    for phase in (["clean", "obstructed"] if request.obstacle_comparison else [None]):
        for simulator in (["mujoco", "gazebo"] if request.sim == "both" else [request.sim]):
            for controller in request.controllers:
                name = simulator if controller == "dwarl" else f"{simulator}-{controller}"
                yield phase, simulator, controller, f"{name}-{phase}" if phase else name


def _tasks(contract, workload, out, request, rows, checkpoint, comparison):
    phase_request, selected, previous = request, workload, None
    for phase, simulator, controller, name in _variants(request):
        if phase and phase != previous:
            phase_request = deepcopy(request)
            phase_request.obstacle_comparison = False
            phase_request.unmapped_obstacles = 0 if phase == "clean" else (request.unmapped_obstacles or 3)
            if phase == "obstructed":
                from dwarl.testing.comparison import freeze_obstacles

                selected, manifest = freeze_obstacles(
                    contract, workload, out, {k: v for k, v in rows.items() if k.endswith("-clean")},
                    phase_request.unmapped_obstacles, request.unmapped_near_doorways, checkpoint)
                comparison.update(layout_identity=manifest["identity"], obstacles_file="obstacles.json",
                                  requested_per_case=manifest["requested_per_case"],
                                  placed_per_case={key: len(v["obstacles"]) for key, v in manifest["cases"].items()})
            OmegaConf.save(phase_request, out / f"request_{phase}.yaml", resolve=True)
            previous = phase
        yield phase, simulator, controller, name, phase_request, selected


def execute(cfg, checkpoint=None):
    from dwarl.cli import ui

    cfg = deepcopy(cfg)
    checkpoint = checkpoint or resolve_checkpoint(resolve_run(cfg.test.run), cfg.test.checkpoint)
    print(f"Selected run={checkpoint.run} iteration={checkpoint.iteration} sha256={checkpoint.sha256}", flush=True)
    out = checkpoint.run / "tests" / f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid4().hex[:8]}"
    out.mkdir(parents=True)
    OmegaConf.save(cfg.test, out / "request.yaml", resolve=True)
    rows, summaries, status = {}, {}, "completed"
    comparison = {"mode": "clean_then_obstructed"} if cfg.test.obstacle_comparison else None
    error, bundle, workload = None, None, None
    try:
        if cfg.test.controller_params:
            source = Path(cfg.test.controller_params)
            snapshot = out / "controller_overrides.yaml"
            copyfile(source, snapshot)
            write_json(out / "controller_tuning_source.json", {"source": str(source), "sha256": sha256(snapshot)})
            cfg.test.controller_params = str(snapshot)
            OmegaConf.save(cfg.test, out / "request.yaml", resolve=True)
        contract = load_contract(checkpoint.run)
        OmegaConf.save(contract, out / "contract.yaml", resolve=True)
        workload = materialize(contract, cfg.test, out)
        bundle = prepare_policy(checkpoint)
        rows = {name: [] for _, _, _, name in _variants(cfg.test)}
        for phase, simulator, controller, name, request, selected in _tasks(
                contract, workload, out, cfg.test, rows, checkpoint.sha256, comparison):
            backend = None
            try:
                if simulator == "mujoco":
                    from dwarl.testing.backends.mujoco import MujocoBackend

                    backend = MujocoBackend(contract, bundle, selected, out, request)
                else:
                    from dwarl.testing.backends.gazebo import GazeboBackend

                    options = {"phase": phase} if phase else {}
                    backend = GazeboBackend(contract, bundle, selected, out, request, controller=controller, **options)
                for index, case in enumerate(selected["cases"], 1):
                    ui.info(f"{name} · case {index}/{len(workload['cases'])} · {case['name']} · {case['id'][:8]}")
                    try:
                        row = backend.execute_case(case)
                    except KeyboardInterrupt:
                        row = {**getattr(backend, "partial", {}), "case_id": case["id"], "status": "interrupted"}
                        row.update(controller=controller, backend=simulator, phase=phase)
                        if phase == "obstructed":
                            row.update(unmapped_obstacles=case["unmapped_obstacles"],
                                       unmapped_placement=case["unmapped_placement"])
                        rows[name].append(persist_episode(out, name, case, row, checkpoint.sha256))
                        raise
                    except Exception as exc:
                        row = {**getattr(backend, "partial", {}),
                               "case_id": case["id"], "status": "infrastructure_failure", "error": str(exc)}
                    row.update(controller=controller, backend=simulator, phase=phase)
                    if phase == "obstructed":
                        row["unmapped_obstacles"] = case["unmapped_obstacles"]
                        row["unmapped_placement"] = case["unmapped_placement"]
                    rows[name].append(persist_episode(out, name, case, row, checkpoint.sha256))
                    write_json(out / "episodes.json", rows)
                    ui.finding("pass" if row.get("outcome") == "success" else "check",
                               f"{case['name']}: {row.get('outcome', row.get('status', 'unknown'))}")
            finally:
                if backend is not None:
                    backend.close()
            summaries[name] = summarize(rows[name], len(workload["cases"]))
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:
        status, error = "infrastructure_failure", str(exc)
    finally:
        from dwarl.runs.provenance import versions

        if status != "interrupted" and any(any(r["status"] != "completed" for r in rr) for rr in rows.values()):
            status = "infrastructure_failure"
        if workload:
            for name, episodes in rows.items():
                summaries[name] = summarize(episodes, len(workload["cases"]))
        if comparison is not None:
            from dwarl.testing.comparison import paired_results

            comparison["controllers"] = paired_results(rows, cfg.test.controllers)
        report = {"schema": "dwarl_results/1", "status": status, "error": error,
                  "suite": cfg.test.suite,
                  "versions": versions(),
                  "checkpoint": checkpoint.sha256, "iteration": checkpoint.iteration,
                  "bundle": str(bundle) if bundle else None, "contract": checkpoint.contract,
                   "workload": workload["identity"] if workload else None, "simulators": summaries,
                   "obstacle_comparison": comparison,
                  "passed": bool(summaries) and status == "completed" and all(r["passed"] for r in summaries.values())}
        write_json(out / "episodes.json", rows)
        write_json(out / "summary.json", report)
        if workload and rows:
            from dwarl.testing.results import write_report

            try:
                print("Rendering episode reports...", flush=True)
                write_report(out, workload, rows, report)
                from dwarl.workflows.report import render_html

                render_html(out)
            except Exception as exc:
                print(f"Report rendering failed (raw results saved): {exc}")
        from dwarl.workflows.report import summarize as show_summary

        show_summary(report)
        ui.info(f"Results: {out}")
        ui.next_step("dwarl", "report", out, "--open")
    return out, report
