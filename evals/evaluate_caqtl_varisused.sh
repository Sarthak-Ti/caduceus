#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=caqtl_varisused
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=3

set -euo pipefail
cd /data1/lesliec/sarthak/caduceus/evals

# Task 3 uses the original GM12878 DNase checkpoint and input.
# Submit --array=4-5 for the GM12878 ATAC and HepG2 DNase best-loss models.
# Add matching [ID] entries below, then include those IDs in --array to test more models.
outputs=(
  [3]="nomlm_maskonly_gm12878_caqtl_varisused.npy"
  [4]="nomlm_maskonly_gm12878atac_bestloss_caqtl_varisused.npy"
  [5]="nomlm_maskonly_hepg2_bestloss_caqtl_varisused.npy"
)
ckpts=(
  [3]="/data1/lesliec/sarthak/caduceus/outputs/2025-04-28/15-26-30-700432/checkpoints/14-val_loss=0.27260.ckpt"
  [4]="/data1/lesliec/sarthak/caduceus/outputs/2026-09-24/09-53-54-581722/checkpoints/11-val_loss=0.05880.ckpt"
  [5]="/data1/lesliec/sarthak/caduceus/outputs/2026-09-23/11-45-50-395363/checkpoints/15-val_loss=0.22884.ckpt"
)
data_paths=(
  [3]="/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz"
  [4]="/data1/lesliec/sarthak/data/alphagenome/processed_bigwigs/gm12878_atac_unmap_only.npz"
  [5]="/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/HepG2_DNase.npz"
)

i=${SLURM_ARRAY_TASK_ID:-3}
if [[ -z "${outputs[$i]:-}" || -z "${ckpts[$i]:-}" || -z "${data_paths[$i]:-}" ]]; then
  echo "No output/checkpoint/data path configured for array task $i." >&2
  exit 1
fi

echo "Scoring European ChromBPNet var.isused: task $i, output ${outputs[$i]}"
echo "Checkpoint ${ckpts[$i]}, data path ${data_paths[$i]}"
pixi run python -u caqtl_onemodel_varisused.py \
  --output "${outputs[$i]}" \
  --ckpt_path "${ckpts[$i]}" \
  --data_path "${data_paths[$i]}" \
  --mask_size 1000 \
  --out_size 500 \
  --pool 1 \
  --ctt_val 6 \
  --load_data \
  "$@"
