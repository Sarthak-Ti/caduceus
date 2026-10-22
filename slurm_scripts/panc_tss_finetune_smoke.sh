#!/bin/bash
#SBATCH --partition=preemptable
#SBATCH --account=preemptable
#SBATCH --qos=preemptable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --job-name=panc_tss_smoke
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out
# Smoke test for PANC RNA fine-tuning (tests/panc_tss_finetune.py): transfer audit, one real
# batch forward/backward, then 16 batches (2 updates) of compiled training per arm.
# Usage: sbatch slurm_scripts/panc_tss_finetune_smoke.sh [init:head ...]   (default: multi_sea:both de_only:both)
set -euo pipefail
T=/data1/lesliec/sarthak/triton_upgrade_test
cd /data1/lesliec/sarthak/caduceus
export PATH="/home/tiwaris2/.pixi/bin:$PATH"
export PIXI_CACHE_DIR=$T/.pixi_cache
export TRITON_CACHE_DIR=$T/triton_cache/modern
export HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=4

arms=("$@")
[[ ${#arms[@]} -eq 0 ]] && arms=(multi_sea:both de_only:both)
for arm in "${arms[@]}"; do
  echo "=== smoke ${arm} ==="
  pixi run --manifest-path $T/pixi.toml python -u tests/panc_tss_finetune.py \
    --init "${arm%%:*}" --head "${arm##*:}" --smoke
done
echo "all smoke tests passed: ${arms[*]}"
