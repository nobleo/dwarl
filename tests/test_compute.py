from types import SimpleNamespace

import pytest

from dwarl.config import compose_config
from dwarl.testing.compute import ComputeProfile, distribution


def test_controller_configuration():
    cfg = compose_config(assignments=["test.sim=gazebo", "test.controllers=[dwarl,mppi,dwb,rpp]",
                                      "test.profile_compute=true"])
    assert list(cfg.test.controllers) == ["dwarl", "mppi", "dwb", "rpp"]
    for assignment in ("test.controllers=[mppi]", "test.controllers=[rpp]", "test.controllers=[teb]",
                       "test.controllers=[dwarl,dwarl]", "test.profile_compute=true"):
        with pytest.raises(ValueError):
            compose_config(assignments=[assignment])


def test_compute_missing_and_budget_metrics():
    assert distribution([]) is None
    profile = ComputeProfile(0, .1)
    for value in (5, 20, 150):
        message = SimpleNamespace(status=[SimpleNamespace(values=[
            SimpleNamespace(key="controller_ms", value=str(value)), SimpleNamespace(key="failed", value="0")])])
        profile.timing(message, 0.)
    result = profile.result(1.)
    assert result["calls_over_budget"] == 1
    assert result["controller_ms"]["count"] == 3
    assert result["inference_ms"] is None
    assert result["processes"]["controller_server"]["cpu_percent"] is None


def test_standalone_model_replay_and_model_identity(tmp_path):
    import numpy as np
    import onnx
    import yaml
    from onnx import TensorProto, helper

    from dwarl.runs.store import sha256
    from dwarl.testing.model_benchmark import execute

    graph = helper.make_graph([helper.make_node("Identity", ["obs"], ["actions"])], "replay",
                              [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 2])],
                              [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.save(model, tmp_path / "policy.onnx")
    (tmp_path / "deploy.yaml").write_text(yaml.safe_dump({
        "control": {"dt": .1}, "model": {"inputs": [{"name": "obs", "group": "state", "shape": [1, 2]}]}}))
    observations = tmp_path / "observations.npz"
    np.savez(observations, state=np.ones((3, 2), dtype=np.float32), model_sha256=sha256(tmp_path / "policy.onnx"))
    report = execute(tmp_path, observations, tmp_path / "report.json", iterations=10, warmup=2)
    assert report["recorded_samples"] == 3
    assert report["inference_ms"]["count"] == 10
    np.savez(observations, state=np.ones((3, 2), dtype=np.float32), model_sha256="wrong")
    with pytest.raises(ValueError, match="different ONNX"):
        execute(tmp_path, observations, tmp_path / "report.json", iterations=10)
