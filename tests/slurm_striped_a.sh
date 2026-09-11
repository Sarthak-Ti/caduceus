#!/bin/bash
#SBATCH --partition=preemptable
#SBATCH --account=preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --job-name=striped_a
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Part A: per-block scaling (locates the O(L^2) attention wall and compares one Hydra
# block against one Caduceus BiMamba layer) + the core head-to-head at production shape.
source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi -L; mkdir -p tests/results

pixi run python -u -m tests.bench_striped --part blocks \
    --block-seqlens 32768 65536 131072 262144 524288 --iters 3 --warmup 2 \
    --out tests/results/striped_blocks.json || echo "BLOCKS FAILED"

pixi run python -u -m tests.bench_striped --part models --configs core \
    --seqlens 131072 262144 524288 --iters 3 --warmup 2 \
    --out tests/results/striped_core.json || echo "CORE FAILED"
echo "=== done ==="
