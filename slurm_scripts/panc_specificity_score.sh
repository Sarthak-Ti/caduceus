#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=panc_spec_score
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out
# Score every prediction file in a directory (evals/panc_specificity.py score), twice:
# all 13 cell types (with the specificity metrics) and DE only (plain across-gene evaluation).
# Usage: sbatch slurm_scripts/panc_specificity_score.sh <pred_dir> [out_prefix]
set -euo pipefail
T=/data1/lesliec/sarthak/triton_upgrade_test
cd /data1/lesliec/sarthak/caduceus
export PATH="/home/tiwaris2/.pixi/bin:$PATH"
export PIXI_CACHE_DIR=$T/.pixi_cache

dir=${1:?Specify the prediction directory}
prefix=${2:-$dir/scores}
shopt -s nullglob
preds=("$dir"/*.npz)
[[ ${#preds[@]} -gt 0 ]] || { echo "no prediction files in $dir" >&2; exit 1; }
echo "scoring ${#preds[@]} files: ${preds[*]}"
for p in "${preds[@]}"; do
  pixi run --manifest-path $T/pixi.toml python -c "import numpy as np; print('$p', np.load('$p')['ckpt'])" 2>/dev/null | tail -1
done
pixi run --manifest-path $T/pixi.toml python -u evals/panc_specificity.py score "${preds[@]}" --out "${prefix}_all13" --full
pixi run --manifest-path $T/pixi.toml python -u evals/panc_specificity.py score "${preds[@]}" --out "${prefix}_DE" --celltypes DE
