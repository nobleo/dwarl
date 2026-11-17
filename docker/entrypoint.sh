#!/bin/bash
# Activate the locked Pixi environment and its built overlays, then run the command in /work.
set -e
export HOME="${HOME:-/tmp/home}"
mkdir -p "$HOME" 2>/dev/null || true
set +u
source /opt/dwarl-activate.sh
export PYTHONPATH="/opt/dwarl/demo${PYTHONPATH:+:$PYTHONPATH}"
cd /work
exec "$@"
