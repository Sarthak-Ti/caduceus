#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=e2g_enformer_dn
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-1

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

#-- original accessibility-reduction run, kept for reference --
##SBATCH --array=0-0
# outputs=("k562_unstranded_enformer_tss_centered")
#
# ckpts=( \
#   "/data1/lesliec/sarthak/caduceus/outputs/2025-08-15/12-23-01-318424/checkpoints/last.ckpt" \
# )
#
# i=$SLURM_ARRAY_TASK_ID
# OUTPUT=${outputs[$i]}
# CKPT=${ckpts[$i]}
#
# pixi run python -u e2g_enformer.py \
#   -o "$OUTPUT" \
#   --ckpt_path "$CKPT" \
#   --scale_factor 100 \
#   --dist_additional_mask 100 \
#   --load_data \
#   --center_tss

#-- two jobs: dinucleotide shuffle alone, and shuffle + accessibility reduction --
#new output names so the accessibility-reduction results above are not overwritten
outputs=("k562_unstranded_enformer_tss_centered_dinuc" "k562_unstranded_enformer_tss_centered_dinuc_acc")
perturbations=("shuffle" "both")

CKPT="/data1/lesliec/sarthak/caduceus/outputs/2025-08-15/12-23-01-318424/checkpoints/last.ckpt"
N_SHUFFLES=11

#-- pick the right one based on SLURM_ARRAY_TASK_ID --
i=$SLURM_ARRAY_TASK_ID
OUTPUT=${outputs[$i]}
PERTURBATION=${perturbations[$i]}

echo "Running task $i: output=$OUTPUT, ckpt=$CKPT, perturbation=$PERTURBATION, n_shuffles=$N_SHUFFLES"
#-- run the task --
#--scale_factor and --dist_additional_mask are inert for perturbation=shuffle, only used by both
pixi run python -u e2g_enformer_dn.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --perturbation "$PERTURBATION" \
  --n_shuffles "$N_SHUFFLES" \
  --dist_additional_shuffle 100 \
  --scale_factor 100 \
  --dist_additional_mask 100 \
  --seed 0 \
  --batch_size 2 \
  --load_data \
  --center_tss
