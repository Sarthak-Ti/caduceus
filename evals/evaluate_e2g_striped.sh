#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=150G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=e2g_borzoi
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-1

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

outputs=("k562_striped_tss_centered" "k562_transformer_tss_centered")

ckpts=( \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-05-13/12-20-24-265445/checkpoints/02-step=19250.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-05-10/13-08-03-382462/checkpoints/00-val_loss=-0.02792.ckpt" \
)


#-- pick the right one based on SLURM_ARRAY_TASK_ID --
i=$SLURM_ARRAY_TASK_ID
OUTPUT=${outputs[$i]}
CKPT=${ckpts[$i]}
ASSAY=${assay_types[$i]}

echo "Running task $i: output=$OUTPUT, ckpt=$CKPT, assay=$ASSAY"
#-- run the task --
pixi run python -u e2g_striped.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --scale_factor 100 \
  --dist_additional_mask 100 \
  --data_path /data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/K562_DNase.npz \
  --ctt_val 1 \
  --pool 32 \
  --center_tss \
