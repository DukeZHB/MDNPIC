#!/usr/bin/env bash
# Train MDNPIC on the mhist dataset
set -e

export EXP_DIR=./results_mhist
export N_SPLITS=1
export DEVICE_ID=0

python main.py \
    --device ${DEVICE_ID} \
    --loss mdnpic \
    --config configs/mhist.yml \
    --exp ${EXP_DIR} \
    --doc mhist \
    --n_splits ${N_SPLITS} \
    --ni
