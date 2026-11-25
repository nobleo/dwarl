#!/bin/bash
# Activate the locked Pixi environment and its built overlays, then run the command in /work.
set -e
mkdir -p "$HOME"
source /opt/dwarl-activate.sh
cd /work
if [[ ${1:-} == demo ]]; then
  # Same as the `demo` task in pixi.toml.
  data=(--set train.data.train_maps=10 --set train.data.validation_maps=2 --set train.data.test_maps=1
        --set train.data.pairs_per_map=3)
  dwarl setup --example --yes
  exec dwarl train diffbot --preset navigation --show --num-envs 10 --set train.validation_cases=2 "${data[@]}" \
    --set recipe.sim.realtime=true
fi
exec dwarl "$@"
