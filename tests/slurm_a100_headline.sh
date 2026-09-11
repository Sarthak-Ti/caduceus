#!/bin/bash
#SBATCH --partition=gpu,lesliec
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --job-name=mamba_a100
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Headline baseline-vs-optimized numbers on the GPU the production job actually uses.
# Deliberately small and short so it backfills: every A100 in the cluster is currently
# 4/4 allocated, and a 30-minute job fits gaps that a 2-hour one does not.
source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi -L; mkdir -p tests/results

echo "########## backbone at production shape ##########"
for V in baseline ckpt; do
  pixi run python -m tests.bench_mamba --mode backbone --variant "${V}" \
      --seqlens 262144 524288 --iters 3 --warmup 2 \
      --out "tests/results/a100_backbone_${V}.json" || echo "${V} backbone FAILED"
done

echo "########## full training step (B=1) and with checkpointing at B=2 ##########"
pixi run python -m tests.bench_mamba --mode full --variant baseline --seqlens 524288 \
    --iters 3 --warmup 2 --out "tests/results/a100_full_baseline.json" || echo "full baseline FAILED"
for B in 1 2; do
  pixi run python -m tests.bench_mamba --mode full --variant ckpt --seqlens 524288 \
      --batch-size ${B} --iters 3 --warmup 2 \
      --out "tests/results/a100_full_ckpt_b${B}.json" || echo "full ckpt b${B} FAILED"
done

echo "########## inference ##########"
pixi run python -m tests.bench_mamba --mode infer --variant baseline --seqlens 524288 \
    --infer-batch-sizes 1 2 4 --iters 3 --warmup 2 \
    --out "tests/results/a100_infer_baseline.json" || echo "infer FAILED"
echo "=== done ==="
