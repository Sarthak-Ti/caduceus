#!/bin/bash

#SBATCH --partition=lesliec,gpu,gpushort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=eqtl_benchmark
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-1

# 4 tasks: IDs 0,1,2,3 if 0-3

# source ~/.bashrc
cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi


outputs=("striped" "transformer" "nopretrain" "immune" "nobcell" "nobcell_nomlm_maskonly")

ckpts=( \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-05-13/12-20-24-265445/checkpoints/02-step=19250.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-05-10/13-08-03-382462/checkpoints/00-val_loss=-0.02792.ckpt" \
)

#–– pick the right one based on SLURM_ARRAY_TASK_ID ––
i=$SLURM_ARRAY_TASK_ID
OUTPUT=${outputs[$i]}
CKPT=${ckpts[$i]}

echo "Running task $i with output $OUTPUT, ckpt $CKPT"
#–– run the ask ––
pixi run python -u eqtl_onemodel_striped.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --data_path /data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz \
  --ctt_val 4 \
#   --load_data
#data idxs is only if trained on multiple cell types