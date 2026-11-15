"""train.init_from: a new run starts from another run's weights where shapes match."""

import torch


def test_warm_start_copies_matching_weights_but_not_exploration_noise(tmp_path):
    from types import SimpleNamespace

    from dwarl.training.checkpoint import warm_start

    def model(inputs):
        net = torch.nn.Module()
        net.layer = torch.nn.Linear(inputs, 2)
        net.register_buffer("obs_normalizer_std", torch.ones(inputs))  # a normalizer statistic named *std
        net.distribution = torch.nn.Module()
        net.distribution.std_param = torch.nn.Parameter(torch.full((2,), 0.6))
        return net

    old_actor, old_critic = model(3), model(5)
    for net in (old_actor, old_critic):
        torch.nn.init.constant_(net.layer.weight, 7.0)
        net.obs_normalizer_std.fill_(3.0)
        net.distribution.std_param.data.fill_(0.1)
    torch.save({"actor_state_dict": old_actor.state_dict(), "critic_state_dict": old_critic.state_dict()},
               tmp_path / "model.pt")
    actor, critic = model(3), model(6)  # the critic gained inputs
    runner = SimpleNamespace(alg=SimpleNamespace(_raw_actor=actor, _raw_critic=critic))
    warm_start(runner, tmp_path / "model.pt", "cpu")
    assert (actor.layer.weight == 7).all() and (actor.obs_normalizer_std == 3).all()
    assert (actor.distribution.std_param == 0.6).all()      # fresh exploration noise
    assert not (critic.layer.weight == 7).all()             # shape changed: fresh layer


def test_init_checkpoint_is_validated():
    import pytest

    from dwarl.config import compose_config

    assert compose_config(assignments=["train.init_checkpoint=350"]).train.init_checkpoint == "350"
    with pytest.raises(ValueError, match="init_checkpoint"):
        compose_config(assignments=["train.init_checkpoint=newest"])
