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


def execute(cfg, checkpoint=None):
    from dwarl.cli import ui

    cfg = deepcopy(cfg)
    checkpoint = checkpoint or resolve_checkpoint(resolve_run(cfg.test.run), cfg.test.checkpoint)
    print(f"Selected run={checkpoint.run} iteration={checkpoint.iteration} sha256={checkpoint.sha256}", flush=True)
    out = checkpoint.run / "tests" / f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid4().hex[:8]}"
    out.mkdir(parents=True)
    OmegaConf.save(cfg.test, out / "request.yaml", resolve=True)
    rows, summaries, status = {}, {}, "completed"
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
        variants = [(sim, controller) for sim in
                    (["mujoco", "gazebo"] if cfg.test.sim == "both" else [cfg.test.sim])
                    for controller in cfg.test.controllers]
        for simulator, controller in variants:
            name = simulator if controller == "dwarl" else f"{simulator}-{controller}"
            rows[name] = []
            backend = None
            try:
                if simulator == "mujoco":
                    from dwarl.testing.backends.mujoco import MujocoBackend

                    backend = MujocoBackend(contract, bundle, workload, out, cfg.test)
                else:
                    from dwarl.testing.backends.gazebo import GazeboBackend

                    backend = GazeboBackend(contract, bundle, workload, out, cfg.test, controller=controller)
                for index, case in enumerate(workload["cases"], 1):
                    ui.info(f"{name} · case {index}/{len(workload['cases'])} · {case['name']} · {case['id'][:8]}")
                    try:
                        row = backend.execute_case(case)
                    except KeyboardInterrupt:
                        row = {**getattr(backend, "partial", {}), "case_id": case["id"], "status": "interrupted"}
                        row.update(controller=controller, backend=simulator)
                        rows[name].append(persist_episode(out, name, case, row, checkpoint.sha256))
                        raise
                    except Exception as exc:
                        row = {**getattr(backend, "partial", {}),
                               "case_id": case["id"], "status": "infrastructure_failure", "error": str(exc)}
                    row.update(controller=controller, backend=simulator)
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
        report = {"schema": "dwarl_results/1", "status": status, "error": error,
                  "suite": cfg.test.suite,
                  "versions": versions(),
                  "checkpoint": checkpoint.sha256, "iteration": checkpoint.iteration,
                  "bundle": str(bundle) if bundle else None, "contract": checkpoint.contract,
                  "workload": workload["identity"] if workload else None, "simulators": summaries,
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
