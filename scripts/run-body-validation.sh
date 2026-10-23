#!/usr/bin/env bash
# Synthetic-only integration check. Suitable for a transient user service so
# long offline inference survives tool/session interruptions. No camera opens.
set -euo pipefail
# Fail closed after a host freeze during validation. Do not rerun this workload
# until its worker has tested RAM/swap limits and the user authorizes another run.
printf '%s\n' 'Full-body validation disabled after host memory-pressure incident. No inference started.' >&2
exit 1
cd "$(dirname "$0")/.."
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export XDG_CONFIG_HOME="$XDG_CACHE_HOME/deepfake/validation-config"
export XDG_STATE_HOME="$XDG_CACHE_HOME/deepfake/validation-state"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
worker_python="${DEEPFAKE_BODY_PYTHON:-$HOME/.venvs/deepfake-editalive/bin/python}"
worker_checkout="${DEEPFAKE_BODY_CHECKOUT:-$XDG_CACHE_HOME/deepfake/vendor/EditaLive}"
python scripts/prepare-body-fixture.py
python -m deepfake experimental configure --backend editalive --checkout "$worker_checkout" --python "$worker_python"
DEEPFAKE_CONSENT_ACK=1 python -m deepfake video \
  -i /tmp/deepfake-integration-artifacts/body-input.mp4 \
  -o /tmp/deepfake-integration-artifacts/body-output.mp4 \
  --engine experimental --mode fullbody --model-backend editalive \
  --reference docs/demos/fullbody-reference.png --frame-gen off --no-widget --no-preview \
  --encoder libx264 --prompt 'Transform the person into the reference character with short auburn hair wearing a teal technical flight jumpsuit with silver panels. Preserve the full body pose and studio background.'
