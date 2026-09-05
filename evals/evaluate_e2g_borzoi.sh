#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=e2g_borzoi_dn
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-1

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

#-- original accessibility-reduction sweep over all three assay types, kept for reference --
##SBATCH --array=0-2
# outputs=("k562_stranded_borzoi_cage_tss_centered" "k562_stranded_borzoi_rnaseq_tss_centered" "k562_stranded_borzoi_both_tss_centered")
#
# ckpts=( \
#   "/data1/lesliec/sarthak/caduceus/outputs/2026-02-25/23-21-56-969652/checkpoints/last.ckpt" \
#   "/data1/lesliec/sarthak/caduceus/outputs/2026-02-25/23-33-18-468516/checkpoints/last.ckpt" \
#   "/data1/lesliec/sarthak/caduceus/outputs/2026-02-25/23-37-06-964773/checkpoints/last.ckpt" \
# )
#
# assay_types=("CAGE" "RNA" "both")
#
# i=$SLURM_ARRAY_TASK_ID
# OUTPUT=${outputs[$i]}
# CKPT=${ckpts[$i]}
# ASSAY=${assay_types[$i]}
#
# pixi run python -u e2g_borzoi.py \
#   -o "$OUTPUT" \
#   --ckpt_path "$CKPT" \
#   --assay_type "$ASSAY" \
#   --scale_factor 100 \
#   --dist_additional_mask 100 \
#   --load_data \
#   --center_tss

#-- CAGE model only, two jobs: dinucleotide shuffle alone, and shuffle + accessibility reduction --
#new output names so the accessibility-reduction results above are not overwritten
outputs=("k562_stranded_borzoi_cage_tss_centered_dinuc" "k562_stranded_borzoi_cage_tss_centered_dinuc_acc")
perturbations=("shuffle" "both_perturbations")

CKPT="/data1/lesliec/sarthak/caduceus/outputs/2026-02-25/23-21-56-969652/checkpoints/last.ckpt"
ASSAY="CAGE"
N_SHUFFLES=11

#-- pick the right one based on SLURM_ARRAY_TASK_ID --
i=$SLURM_ARRAY_TASK_ID
OUTPUT=${outputs[$i]}
PERTURBATION=${perturbations[$i]}

echo "Running task $i: output=$OUTPUT, ckpt=$CKPT, assay=$ASSAY, perturbation=$PERTURBATION, n_shuffles=$N_SHUFFLES"
#-- run the task --
#--scale_factor and --dist_additional_mask are inert for perturbation=shuffle, only used by both_perturbations
pixi run python -u e2g_borzoi_dn.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --assay_type "$ASSAY" \
  --perturbation "$PERTURBATION" \
  --n_shuffles "$N_SHUFFLES" \
  --dist_additional_shuffle 100 \
  --scale_factor 100 \
  --dist_additional_mask 100 \
  --seed 0 \
  --batch_size 2 \
  --load_data \
  --center_tss
