"""Presets, config files and saved runs may add manager terms; the manager schemas validate them."""

import pytest
from omegaconf import OmegaConf

from dwarl.config import compose_config


def test_config_file_and_saved_run_may_add_terms(tmp_path):
    source = tmp_path / "dwarl.yaml"
    OmegaConf.save({"recipe": {
        "rewards": {"bonus": {"func": "dwarl.navigation.rewards:time", "weight": -0.1}},
        "observations": {"privileged": {"terms": {"extra": {"func": "dwarl.navigation.observations:last_action"}}}},
    }}, source)
    cfg = compose_config(config=source)
    assert cfg.recipe.rewards.bonus.weight == -0.1 and "extra" in cfg.recipe.observations.privileged.terms
    saved = tmp_path / "config.yaml"
    OmegaConf.save(cfg, saved, resolve=True)  # what validation and `dwarl test` load from a run
    assert "bonus" in compose_config(config=saved).recipe.rewards


def test_added_terms_are_still_validated(tmp_path):
    source = tmp_path / "dwarl.yaml"
    OmegaConf.save({"recipe": {"rewards": {"bad": {"func": "not a reference", "weight": 1.0}}}}, source)
    with pytest.raises(ValueError, match="package.module:symbol"):
        compose_config(config=source)


def test_preset_may_carry_recipe_settings(monkeypatch):
    from dwarl.config import presets

    monkeypatch.setitem(presets.PRESETS, "probe", {"name": "probe", "recipe": {
        "rewards": {"time": {"weight": 2.0}}}})
    cfg = compose_config(preset="probe")
    assert cfg.train.name == "probe" and cfg.recipe.rewards.time.weight == 2.0
