#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=e2g_benchmark
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-0

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

outputs=("k562_unstranded_enformer_tss_centered")

ckpts=( \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-08-15/12-23-01-318424/checkpoints/last.ckpt" \
)

#-- pick the right one based on SLURM_ARRAY_TASK_ID --
i=$SLURM_ARRAY_TASK_ID
OUTPUT=${outputs[$i]}
CKPT=${ckpts[$i]}

echo "Running task $i with output $OUTPUT, ckpt $CKPT"
#-- run the task --
pixi run python -u e2g_enformer.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --scale_factor 100 \
  --dist_additional_mask 100 \
  --load_data \
  --center_tss \
