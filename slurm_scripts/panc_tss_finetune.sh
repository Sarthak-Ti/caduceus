#!/bin/bash
#SBATCH --partition=lesliec,gpu
#SBATCH --account=lesliec
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --job-name=panc_ft
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out
# PANC RNA fine-tuning, phase 1 (claude_summaries/panc_finetune_plan_20261020.md).
# Config, transfer and resume logic: tests/panc_tss_finetune.py. Outputs in outputs/panc_finetune/<init>_<head>/.
# Usage: sbatch --job-name=panc_ft_<init>_<head> slurm_scripts/panc_tss_finetune.sh <init> <head> [epochs]
#   init: de_only | multi_sea      head: count | profile | both
# A requeued or resubmitted job resumes from the newer of checkpoints/last.ckpt and recovery/last.ckpt.
set -euo pipefail
T=/data1/lesliec/sarthak/triton_upgrade_test
cd /data1/lesliec/sarthak/caduceus
export PATH="/home/tiwaris2/.pixi/bin:$PATH"
export PIXI_CACHE_DIR=$T/.pixi_cache
export TRITON_CACHE_DIR=$T/triton_cache/modern
export HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=4

init=${1:?Specify init: de_only | multi_sea}
head=${2:?Specify head: count | profile | both}
epochs=()
[[ $# -ge 3 ]] && epochs=(--epochs "$3")

exec pixi run --manifest-path $T/pixi.toml python -u tests/panc_tss_finetune.py \
  --init "$init" --head "$head" "${epochs[@]}"
