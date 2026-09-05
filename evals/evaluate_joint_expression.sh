#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00
#SBATCH --mem=150G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=eval_expression_borzoi
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-1

source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

# GM12878 CAGE finetune on Borzoi data (2 strands, 32 bp bins, 6144 bins over 196608 bp).
# Targets are already binned, so no spatial pooling is wanted -> POOL=1.
# (POOL=4 would view them at 128 bp, matching the Enformer-data baseline.)
RUN=/data1/lesliec/sarthak/caduceus/outputs/2026-08-11/19-22-12-155698/checkpoints
EP4_CKPT="$RUN/04-val_loss=0.02610.ckpt"
EP10_CKPT="$RUN/10-val_loss=0.02781.ckpt"

# array index -> (ckpt, pool, out_name)
# 0: epoch 4  (best val/loss, 0.02610)   pool=1
# 1: epoch 10 (0.02781)                  pool=1

case $SLURM_ARRAY_TASK_ID in
    0) CKPT="$EP4_CKPT"  ; POOL=1 ; OUT=borzoi_gm12878_cage_ep4  ;;
    1) CKPT="$EP10_CKPT" ; POOL=1 ; OUT=borzoi_gm12878_cage_ep10 ;;
esac

pixi run python eval_joint_expression.py \
    --ckpt_path "$CKPT" \
    --skip_softplus \
    --save_outputs \
    --save_targets \
    --pool $POOL \
    --num_workers 8 \
    --out_name $OUT