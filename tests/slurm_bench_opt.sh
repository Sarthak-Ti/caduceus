#!/bin/bash
#SBATCH --partition=gpu,lesliec
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=mamba_bench_opt
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Head-to-head of every optimization variant at the production shape, plus the
# batch-size scaling that gradient checkpointing is supposed to unlock.
source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-none} ==="
nvidia-smi
mkdir -p tests/results

PROD_L=524288

for V in baseline fused_bidir ckpt fused_bidir_ckpt; do
  echo
  echo "############ VARIANT ${V}  (B=1, sweep) ############"
  pixi run python -m tests.bench_mamba --mode backbone --variant "${V}" \
      --seqlens 65536 131072 262144 524288 \
      --out "tests/results/backbone_${V}.json" || echo "variant ${V} backbone FAILED"

  echo
  echo "############ VARIANT ${V}  full step (B=1, L=${PROD_L}) ############"
  pixi run python -m tests.bench_mamba --mode full --variant "${V}" \
      --seqlens ${PROD_L} \
      --out "tests/results/full_${V}.json" || echo "variant ${V} full FAILED"
done

# Checkpointing's payoff is that it makes B>1 fit; measure whether the extra
# recompute is bought back by the better-filled selective-scan grid.
for B in 2 4; do
  for V in ckpt fused_bidir_ckpt; do
    echo
    echo "############ VARIANT ${V}  full step (B=${B}, L=${PROD_L}) ############"
    pixi run python -m tests.bench_mamba --mode full --variant "${V}" \
        --seqlens ${PROD_L} --batch-size ${B} \
        --out "tests/results/full_${V}_b${B}.json" || echo "variant ${V} B=${B} FAILED"
  done
done

echo "=== done ==="
