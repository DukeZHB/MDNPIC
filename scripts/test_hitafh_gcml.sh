#!/usr/bin/env bash
# Test MDNPIC on the hitafh_gcml dataset (best checkpoint)
set -e

export EXP_DIR=./results_hitafh_gcml
export DEVICE_ID=0

python main.py \
    --device ${DEVICE_ID} \
    --loss mdnpic \
    --config ${EXP_DIR}/logs/ \
    --exp ${EXP_DIR} \
    --doc hitafh_gcml \
    --test \
    --eval_best
