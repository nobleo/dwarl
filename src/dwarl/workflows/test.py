import time
from uuid import uuid4

from omegaconf import OmegaConf

from dwarl.artifacts.prepare import prepare_policy
from dwarl.runs.store import resolve_checkpoint, resolve_run, write_json
from dwarl.testing.cases import materialize
from dwarl.testing.scoring import summarize
from dwarl.training.contract import load_contract


def execute(cfg, checkpoint=None):
    checkpoint = checkpoint or resolve_checkpoint(resolve_run(cfg.test.run), cfg.test.checkpoint)
    print(f"Selected run={checkpoint.run} iteration={checkpoint.iteration} sha256={checkpoint.sha256}", flush=True)
    out = checkpoint.run / "tests" / f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid4().hex[:8]}"
    out.mkdir(parents=True)
    OmegaConf.save(cfg.test, out / "request.yaml", resolve=True)
    rows, summaries, status = {}, {}, "completed"
    error, bundle, workload = None, None, None
    try:
        contract = load_contract(checkpoint.run)
        OmegaConf.save(contract, out / "contract.yaml", resolve=True)
        workload = materialize(contract, cfg.test, out)
        bundle = prepare_policy(checkpoint)
        for name in (["mujoco", "gazebo"] if cfg.test.sim == "both" else [cfg.test.sim]):
            rows[name] = []
            backend = None
            try:
                if name == "mujoco":
                    from dwarl.testing.backends.mujoco import MujocoBackend

                    backend = MujocoBackend(contract, bundle, workload, out, cfg.test)
                else:
                    from dwarl.testing.backends.gazebo import GazeboBackend

                    backend = GazeboBackend(contract, bundle, workload, out, cfg.test)
                for case in workload["cases"]:
                    try:
                        rows[name].append(backend.execute_case(case))
                    except KeyboardInterrupt:
                        rows[name].append({**getattr(backend, "partial", {}), "case_id": case["id"],
                                           "status": "interrupted"})
                        raise
                    except Exception as exc:
                        rows[name].append({**getattr(backend, "partial", {}),
                                           "case_id": case["id"], "status": "infrastructure_failure",
                                           "error": str(exc)})
                    write_json(out / "episodes.json", rows)
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
                write_report(out, workload, rows, report)
            except Exception as exc:
                print(f"Report rendering failed (raw results saved): {exc}")
        print(f"Test results: {out} ({status})", flush=True)
    return out, report
