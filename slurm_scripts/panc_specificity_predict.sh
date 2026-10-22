#!/bin/bash
#SBATCH --partition=lesliec,gpu
#SBATCH --account=lesliec
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --job-name=panc_spec_predict
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out
# Predictions for the PANC specificity benchmark (evals/panc_specificity.py predict).
# Usage: sbatch slurm_scripts/panc_specificity_predict.sh <ckpt> <out.npz> [predict options, e.g. --max_genes 20 --trans_row_override 3]
set -euo pipefail
T=/data1/lesliec/sarthak/triton_upgrade_test
cd /data1/lesliec/sarthak/caduceus
export PATH="/home/tiwaris2/.pixi/bin:$PATH"
export PIXI_CACHE_DIR=$T/.pixi_cache
export TRITON_CACHE_DIR=$T/triton_cache/modern
export OMP_NUM_THREADS=4

ckpt=${1:?Specify checkpoint}
out=${2:?Specify output npz}
shift 2
exec pixi run --manifest-path $T/pixi.toml python -u evals/panc_specificity.py predict \
  --ckpt "$ckpt" --out "$out" "$@"
