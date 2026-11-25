# Datagen

`setup` selects the robot. `datagen` prepares its training, validation and held-out test maps without starting training:

```bash
pixi run datagen --preset navigation
```

Use the same `--preset`, `--seed`, `--config` and [data overrides](configuration.md) as `train`. For a tiny dataset, use `--preset smoke`.

This step is optional: on a cache miss, `train` prints the matching `dwarl datagen` command and generates the maps. Later runs reuse them.
