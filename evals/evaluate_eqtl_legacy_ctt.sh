#!/bin/bash

#SBATCH --partition=lesliec,gpu,gpushort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=eqtl_benchmark
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-0

# 4 tasks: IDs 0,1,2,3 if 0-3

# source ~/.bashrc
cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi


outputs=("ctt_legacy")

ckpts=( \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-10-28/13-01-43-325458/checkpoints/last.ckpt" \
)

#–– pick the right one based on SLURM_ARRAY_TASK_ID ––
i="${SLURM_ARRAY_TASK_ID:-0}"
OUTPUT=${outputs[$i]}
CKPT=${ckpts[$i]}

echo "Running task $i with output $OUTPUT, ckpt $CKPT"
#–– run the ask ––
pixi run python eqtl_onemodel_ctt_legacy.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --data_path /data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz \
  --ctt_val 6 \
  --load_data
#data idxs is only if trained on multiple cell types