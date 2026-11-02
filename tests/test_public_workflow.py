import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from dwarl.config import compose_config
from dwarl.runs.store import complete_checkpoint, resolve_checkpoint, resolve_run, write_json
from dwarl.training.runner_adapter import RunnerAdapter


def test_yaml_cli_equivalence_and_path_origins(tmp_path):
    configs = tmp_path / "experiments"
    configs.mkdir()
    (configs / "run.yaml").write_text("robot: ../robot\ntest:\n  sim: gazebo\ntrain:\n  num_envs: 8\n")
    yaml = compose_config("experiments/run.yaml", cwd=tmp_path)
    cli = compose_config(assignments=["robot=./robot", "test.sim=gazebo", "train.num_envs=8"], cwd=tmp_path)
    assert OmegaConf.to_container(yaml) == OmegaConf.to_container(cli)
    (tmp_path / "dwarl.yaml").write_text("train:\n  seed: 123\n")
    assert compose_config("experiments/run.yaml", cwd=tmp_path).train.seed == 42
    assert compose_config(cwd=tmp_path).train.seed == 123


def test_unknown_and_duplicate_keys(tmp_path):
    with pytest.raises(ValueError, match="suggestions.*test.sim"):
        compose_config(assignments=["test.smi=both"], cwd=tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        compose_config(assignments=["test.sim=both", "test.sim=gazebo"], cwd=tmp_path)
    (tmp_path / "dwarl.yaml").write_text("train:\n  num_envs: invalid\n")
    with pytest.raises(Exception, match="Integer"):
        compose_config(cwd=tmp_path)


def test_explicit_yaml_keys_are_retained_for_resume(tmp_path):
    (tmp_path / "dwarl.yaml").write_text("train:\n  max_iterations: 12\n  resume: runs/saved\n")
    keys = []
    cfg = compose_config(cwd=tmp_path, explicit_keys=keys, assignments=["train.device=cpu"])
    assert set(keys) == {"train.max_iterations", "train.resume", "train.device"}
    assert cfg.train.resume == str(tmp_path / "runs/saved")
    with pytest.raises(ValueError, match="duplicate"):
        compose_config(cwd=tmp_path, assignments=["train={max_iterations: 3}", "train.max_iterations=4"])


def make_run(path, created=1):
    path.mkdir(parents=True)
    (path / "config.yaml").write_text("schema: dwarl/1\n")
    (path / "contract.yaml").write_text("contract\n")
    write_json(path / "provenance.json", {"schema": "dwarl_run/1", "created_ns": created})
    write_json(path / "checkpoint_manifest.json", {"schema": "dwarl_checkpoints/1", "completed": [], "best": None})
    return path


def test_latest_does_not_fall_back_and_partial_checkpoint_is_ignored(tmp_path):
    old = make_run(tmp_path / "old")
    checkpoint = old / "model_1.pt"
    checkpoint.write_bytes(b"complete")
    complete_checkpoint(old, checkpoint, 1)
    new = make_run(tmp_path / "new", 2)
    (new / "model_999.pt").write_bytes(b"partial")
    assert resolve_run(root=tmp_path) == new
    with pytest.raises(ValueError, match="no completed"):
        resolve_checkpoint(new)
    with pytest.raises(ValueError, match="no best"):
        resolve_checkpoint(old, "best")
    assert resolve_checkpoint(old, "1").path == checkpoint


def test_repeated_saves_never_mutate_completed_checkpoints(tmp_path):
    run = make_run(tmp_path / "run")
    counter = []

    def save(path, infos):
        counter.append(1)
        Path(path).write_bytes(bytes(counter))

    runner = SimpleNamespace(save=save, current_learning_iteration=0)
    adapter = RunnerAdapter.for_runner(runner)
    adapter.after_save.append(lambda path: complete_checkpoint(run, path, 0))
    runner.save(str(run / "model_0.pt"))
    first = resolve_checkpoint(run)
    runner.save(str(run / "model_0.pt"))
    second = resolve_checkpoint(run)
    assert first.path != second.path
    assert first.path.read_bytes() == b"\x01"
    assert second.path.read_bytes() == b"\x01\x01"


def test_failed_save_does_not_complete_checkpoint(tmp_path):
    run = make_run(tmp_path / "run")

    def save(path, infos):
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    runner = SimpleNamespace(save=save)
    RunnerAdapter.for_runner(runner).after_save.append(lambda path: complete_checkpoint(run, path, 1))
    with pytest.raises(OSError):
        runner.save(str(run / "model_1.pt"))
    assert json.loads((run / "checkpoint_manifest.json").read_text())["completed"] == []
    assert not list(run.glob("*.tmp"))


def test_encoder_dimensions_and_activation_override():
    import torch
    from tensordict import TensorDict

    from dwarl.training.models.lidar_cnn import LidarCNNModel, LidarEncoder

    with pytest.raises(ValueError, match="padding"):
        LidarEncoder(1)
    with pytest.raises(ValueError, match="positive"):
        LidarEncoder(180, latent_dim=0)
    obs = TensorDict({"lidar": torch.zeros(2, 1, 1, 180), "state": torch.zeros(2, 7)}, batch_size=[2])
    model = LidarCNNModel(obs, {"actor": ["lidar", "state"]}, "actor", 2, encoder_cfg={"activation": "relu"})
    assert isinstance(model.cnns["lidar"].conv[1], torch.nn.ReLU)
