#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --job-name=striped_smoke
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi -L; mkdir -p tests/results

pixi run python -u -m tests.bench_striped --part blocks \
    --block-seqlens 8192 16384 --iters 2 --warmup 1 \
    --out tests/results/smoke_striped_blocks.json || echo "BLOCKS SMOKE FAILED"

pixi run python -u -m tests.bench_striped --part models --configs all \
    --seqlens 16384 --iters 2 --warmup 1 \
    --out tests/results/smoke_striped_models.json || echo "MODELS SMOKE FAILED"

for M in mamba_only striped transformer_only; do
  pixi run python -u -m tests.profile_mamba --target striped --mode $M \
      --seqlen 8192 --global-pooling 2 --transformer-pooling 2 \
      --iters 2 --warmup 1 --row-limit 10 || echo "PROFILE $M SMOKE FAILED"
done
echo "=== done ==="
