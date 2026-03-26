#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=e2g_benchmark
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=2-2

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

outputs=("k562_tss_500bp" "k562_tss" "k562_tss_simpledecoder")

ckpts=( \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-03-17/14-58-01-055972/checkpoints/13-val_loss=2.18242.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-03-16/18-19-47-711591/checkpoints/09-val_loss=2.17329.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-03-11/15-12-02-454124/checkpoints/08-val_loss=2.25911.ckpt" \
)

#-- pick the right one based on SLURM_ARRAY_TASK_ID --
i=$SLURM_ARRAY_TASK_ID
OUTPUT=${outputs[$i]}
CKPT=${ckpts[$i]}

echo "Running task $i with output $OUTPUT, ckpt $CKPT"
#-- run the task --
pixi run python -u e2g_tss.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --scale_factor 100 \
  --dist_additional_mask 100 \
  --load_data \
