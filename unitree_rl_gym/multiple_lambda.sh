#!/bin/bash

TASK="go2"
MAX_ITER=2000
LAMBDAS=(0.01 0.1 0.5 0.9 1.0 2.0)
LAMBDAS_MONO=(0.01 0.1 0.5)

cd ~/Athi-26532/ARL_Project/unitree_rl_gym

for lam in "${LAMBDAS[@]}"; do
    for lam_mono in "${LAMBDAS_MONO[@]}"; do
        RUN_NAME="rpi_l${lam}_lm${lam_mono}"
        echo ""
        echo "========================================"
        echo " Starting run: $RUN_NAME"
        echo "========================================"
        python legged_gym/scripts/train.py \
            --task $TASK \
            --headless \
            --run_name $RUN_NAME \
            --max_iterations $MAX_ITER \
            --rpi_lambda $lam \
            --rpi_lambda_mono $lam_mono \
            --rpi_target_violation_rate 0.1
    done
done

echo ""
echo "All runs complete. Launch TensorBoard with:"
echo "tensorboard --logdir logs/rough_go2"