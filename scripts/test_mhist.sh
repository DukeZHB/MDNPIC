#!/usr/bin/env bash
# Test MDNPIC on the mhist dataset (best checkpoint)
set -e

export EXP_DIR=./results_mhist
export DEVICE_ID=0

python main.py \
    --device ${DEVICE_ID} \
    --loss mdnpic \
    --config ${EXP_DIR}/logs/ \
    --exp ${EXP_DIR} \
    --doc mhist \
    --test \
    --eval_best
