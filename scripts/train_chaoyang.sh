#!/usr/bin/env bash
# Train MDNPIC on the chaoyang dataset
set -e

export EXP_DIR=./results_chaoyang
export N_SPLITS=1
export DEVICE_ID=0

python main.py \
    --device ${DEVICE_ID} \
    --loss mdnpic \
    --config configs/chaoyang.yml \
    --exp ${EXP_DIR} \
    --doc chaoyang \
    --n_splits ${N_SPLITS} \
    --ni
