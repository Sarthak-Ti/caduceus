#!/bin/bash
#SBATCH --partition=preemptable
#SBATCH --account=preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --job-name=striped_b
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Part B: what the dormant global_pooling U-Net buys, the hybrid/transformer modes at
# pooling levels where attention is affordable, and per-mode kernel attribution.
source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi -L; mkdir -p tests/results

pixi run python -u -m tests.bench_striped --part models --configs pooling \
    --seqlens 524288 --iters 3 --warmup 2 \
    --out tests/results/striped_pooling.json || echo "POOLING FAILED"

pixi run python -u -m tests.bench_striped --part models --configs hybrid \
    --seqlens 524288 --iters 3 --warmup 2 \
    --out tests/results/striped_hybrid.json || echo "HYBRID FAILED"

echo "########## attribution: mamba_only (production striped config) ##########"
pixi run python -u -m tests.profile_mamba --target striped --mode mamba_only \
    --seqlen 262144 --iters 3 --warmup 2 --row-limit 25 || echo "PROF mamba_only FAILED"

echo "########## attribution: striped gp8/tp8 ##########"
pixi run python -u -m tests.profile_mamba --target striped --mode striped \
    --global-pooling 8 --transformer-pooling 8 \
    --seqlen 262144 --iters 3 --warmup 2 --row-limit 25 || echo "PROF striped FAILED"
echo "=== done ==="
