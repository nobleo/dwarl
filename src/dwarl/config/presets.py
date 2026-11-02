"""Explicit starter recipes; normal defaults do not change."""

PRESETS = {
    "smoke": {"name": "smoke", "preset": "smoke", "num_envs": 2, "max_iterations": 2, "save_interval": 1,
              "validation_interval": 0, "data": {"train_maps": 1, "validation_maps": 1, "test_maps": 1,
                                                   "pairs_per_map": 3}},
    "navigation": {"name": "navigation", "preset": "navigation", "validation_interval": 200},
}
