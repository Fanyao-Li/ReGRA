#!/bin/bash
set -e
cd "$(dirname "$0")"
REGRA_CONFIG_PATH="${1:-config/regra/code.yaml}"
REGRA_PYTHON="${REGRA_PYTHON:-../../env/Malora/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}" "$REGRA_PYTHON" train.py -f "$REGRA_CONFIG_PATH" --regra_config.stage=1 --evaluate=False
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}" "$REGRA_PYTHON" train.py -f "$REGRA_CONFIG_PATH" --regra_config.stage=2
