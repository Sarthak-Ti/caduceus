#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00
#SBATCH --mem=150G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=eval_expression
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-3

source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

STRIPED_CKPT=/data1/lesliec/sarthak/caduceus/outputs/2026-05-13/12-20-24-265445/checkpoints/02-step=19250.ckpt
TRANSFORMER_CKPT=/data1/lesliec/sarthak/caduceus/outputs/2026-05-10/13-08-03-382462/checkpoints/00-val_loss=-0.02792.ckpt

# array index -> (ckpt, pool, out_name)
# 0: striped      pool=128
# 1: striped      pool=1
# 2: transformer  pool=128
# 3: transformer  pool=1

case $SLURM_ARRAY_TASK_ID in
    0) CKPT=$STRIPED_CKPT     ; POOL=128 ; OUT=striped_k562_pool128 ;;
    1) CKPT=$STRIPED_CKPT     ; POOL=1   ; OUT=striped_k562_pool1   ;;
    2) CKPT=$TRANSFORMER_CKPT ; POOL=128 ; OUT=transformer_k562_pool128 ;;
    3) CKPT=$TRANSFORMER_CKPT ; POOL=1   ; OUT=transformer_k562_pool1   ;;
esac

pixi run python eval_joint_expression.py \
    --ckpt_path $CKPT \
    --skip_softplus \
    --pool $POOL \
    --num_workers 8 \
    --out_name $OUT