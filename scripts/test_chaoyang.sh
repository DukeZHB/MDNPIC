#!/usr/bin/env bash
# Test MDNPIC on the chaoyang dataset (best checkpoint)
set -e

export EXP_DIR=./results_chaoyang
export DEVICE_ID=0

python main.py \
    --device ${DEVICE_ID} \
    --loss mdnpic \
    --config ${EXP_DIR}/logs/ \
    --exp ${EXP_DIR} \
    --doc chaoyang \
    --test \
    --eval_best
