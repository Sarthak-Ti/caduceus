#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=k562_tss_bulk_finetune_bp_sum
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# =============================================================================
# Overrides: set only what differs from the base defaults.
# See finetune_joint_k562_tss_base_cmd.sh for all available variables.
# =============================================================================

# (no overrides — all defaults apply)

# =============================================================================
source /data1/lesliec/sarthak/caduceus/slurm_scripts/finetune_joint_k562_tss_base_cmd.sh