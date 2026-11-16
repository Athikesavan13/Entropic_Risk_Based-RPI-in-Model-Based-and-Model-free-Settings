#!/bin/bash

TASK="go2"
MAX_ITER=200
QUANTILES=(16 32 64)

cd ~/Athi-26532/ARL_Project/unitree_rl_gym

for nq in "${QUANTILES[@]}"; do
    RUN_NAME="quantile_nq${nq}"
    echo ""
    echo "========================================"
    echo " Starting run: $RUN_NAME"
    echo "========================================"
    python legged_gym/scripts/train.py \
        --task $TASK \
        --headless \
        --run_name $RUN_NAME \
        --max_iterations $MAX_ITER \
        --num_quantiles $nq
done

echo ""
echo "All runs complete. Launch TensorBoard with:"
echo "tensorboard --logdir logs/rough_go2"