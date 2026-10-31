"""Shared result vocabulary; unavailable measurements stay null."""


def score(case, *, success, collision, timeout, travelled, elapsed, clearance,
           trajectory=None, safety_stops=None, recoveries=None, stuck=None, termination_reason=None):
    import numpy as np

    reference = float(np.linalg.norm(np.diff(case["reference_path"], axis=0), axis=1).sum())
    success = bool(success and not collision)
    outcome = ("collision" if collision else "success" if success else "stuck" if stuck else
               "timeout" if timeout else "aborted" if termination_reason == "nav2_aborted" else "terminated")
    return {"case_id": case["id"], "status": "completed", "success": success,
            "outcome": outcome, "stuck": stuck, "termination_reason": termination_reason,
            "collision": bool(collision), "timeout": bool(timeout), "travelled_distance": travelled,
            "reference_path_length": reference, "completion_time": elapsed,
            "clearance": clearance, "path_efficiency": min(1., reference / max(travelled, 1e-9)) if success else 0.,
            "safety_stops": safety_stops, "recovery_actions": recoveries, "trajectory": trajectory}


def summarize(rows, expected):
    completed = [r for r in rows if r["status"] == "completed"]
    count = len(completed)
    return {"expected": expected, "completed": count,
            "infrastructure_failures": sum(r["status"] == "infrastructure_failure" for r in rows),
            "interrupted": sum(r["status"] == "interrupted" for r in rows),
            "success_rate": sum(r["success"] for r in completed) / count if count else None,
            "collision_rate": sum(r["collision"] for r in completed) / count if count else None,
            "complete": count == expected, "passed": count == expected and all(r["success"] for r in completed)}
