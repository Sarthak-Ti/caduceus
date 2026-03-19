#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=k562_tss_finetune
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%A_%a-%x.out
# !! Update --array to 0-(N-1) where N = number of entries below !!
#SBATCH --array=2-2

# =============================================================================
# NAMES — one per job, always unique (used as wandb run name)
# =============================================================================
NAMES=(
  "k562_tss_bulk_finetune_bp_sum"
  "k562_tss_bulk_finetune_bp_sum_no_alt_tss"
  "k562_tss_bulk_finetune_bp_sum_500bptss"
)

# =============================================================================
# VARIANTS — hydra args appended to the base command for each job.
# These are processed last so they override or extend base defaults.
#   override existing key:  optimizer.lr=5e-5
#   add new key:           +dataset.use_alt_tss=false
#   combine:               optimizer.lr=5e-5 +dataset.use_alt_tss=false
# Leave empty string "" to run with all base defaults.
# !! Must have the same number of entries as NAMES !!
# =============================================================================
VARIANTS=(
  ""
  "+dataset.use_alt_tss=false"
  "+dataset.tss_distance=500"
)

# =============================================================================
i=$SLURM_ARRAY_TASK_ID
# i=1
export WANDB_NAME=${NAMES[$i]}
export OVERRIDE_ARGS=${VARIANTS[$i]}

source /data1/lesliec/sarthak/caduceus/slurm_scripts/finetune_joint_k562_tss_base_cmd.sh
