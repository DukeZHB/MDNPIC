#!/usr/bin/env bash
# Train MDNPIC on the hitafh_gcml dataset
set -e

export EXP_DIR=./results_hitafh_gcml
export N_SPLITS=1
export DEVICE_ID=0

python main.py \
    --device ${DEVICE_ID} \
    --loss mdnpic \
    --config configs/hitafh_gcml.yml \
    --exp ${EXP_DIR} \
    --doc hitafh_gcml \
    --n_splits ${N_SPLITS} \
    --ni
