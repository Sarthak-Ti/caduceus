#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --job-name=mamba2_cmp
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi -L; mkdir -p tests/results
TAG="m2_$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')"
pixi run python -m tests.bench_mamba2_compare --seqlen 262144 \
    --out "tests/results/${TAG}_262144.json" || echo "FAILED at 262144"
pixi run python -m tests.bench_mamba2_compare --seqlen 131072 \
    --out "tests/results/${TAG}_131072.json" || echo "FAILED at 131072"
echo "=== done ==="
