#!/bin/bash
#SBATCH --partition=lesliec,gpu
#SBATCH --account=lesliec
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --mem=150G
#SBATCH --time=168:00:00
#SBATCH --requeue
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out
# gp128 training-paradigm comparison (claude_summaries/gp128_paradigm_comparison_plan.md),
# modern env + torch.compile. Every action resumes or skips completed work, so requeues are safe.
# Usage: sbatch --job-name=gp128_<action>_<variant> slurm_scripts/gp128_paradigms.sh <config|preflight|train|eval> <variant>
# Variants: m1_joint | m2_seqonly_finetune | m3_acc_only | m4_scratch
set -euo pipefail
T=/data1/lesliec/sarthak/triton_upgrade_test
cd /data1/lesliec/sarthak/caduceus
export PATH="/home/tiwaris2/.pixi/bin:$PATH"
export PIXI_CACHE_DIR=$T/.pixi_cache
export TRITON_CACHE_DIR=$T/triton_cache/modern
export HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=4
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/gp128_paradigms_mpl_${SLURM_JOB_ID:-local}"
exec pixi run --manifest-path $T/pixi.toml python -u -m tests.gp128_paradigms "${1:?action}" --variant "${2:?variant}"
