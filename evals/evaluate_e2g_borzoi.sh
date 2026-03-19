#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=e2g_borzoi
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-2

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

outputs=("k562_stranded_borzoi_cage_tss_centered" "k562_stranded_borzoi_rnaseq_tss_centered" "k562_stranded_borzoi_both_tss_centered")

ckpts=( \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-02-25/23-21-56-969652/checkpoints/last.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-02-25/23-33-18-468516/checkpoints/last.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-02-25/23-37-06-964773/checkpoints/last.ckpt" \
)

assay_types=("CAGE" "RNA" "both")

#-- pick the right one based on SLURM_ARRAY_TASK_ID --
i=$SLURM_ARRAY_TASK_ID
OUTPUT=${outputs[$i]}
CKPT=${ckpts[$i]}
ASSAY=${assay_types[$i]}

echo "Running task $i: output=$OUTPUT, ckpt=$CKPT, assay=$ASSAY"
#-- run the task --
pixi run python -u e2g_borzoi.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --assay_type "$ASSAY" \
  --scale_factor 100 \
  --dist_additional_mask 100 \
  --load_data \
  --center_tss \
