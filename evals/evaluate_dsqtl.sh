#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=dsqtl_benchmark
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-1

# 4 tasks: IDs 0,1,2,3 if 0-3

# source ~/.bashrc
cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi


outputs=("immune_nob.npy" "immune_all.npy")

ckpts=( \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-04-17/12-31-41-192495/checkpoints/09-val_loss=1.10646.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-04-17/12-29-49-150674/checkpoints/06-val_loss=1.12144.ckpt" \
)

mask_sizes=(1000 1000)

#–– pick the right one based on SLURM_ARRAY_TASK_ID ––
i=$SLURM_ARRAY_TASK_ID
OUTPUT=${outputs[$i]}
CKPT=${ckpts[$i]}
MASK_SIZE=${mask_sizes[$i]}

echo "Running task $i with output $OUTPUT, ckpt $CKPT, mask size $MASK_SIZE"
#–– run the task ––
pixi run python dsqtl_onemodel.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --mask_size "$MASK_SIZE" \
  --load_data
#data idxs is only if trained on multiple cell types