#!/bin/bash
#SBATCH --partition=gpu,lesliec
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=mamba_tests
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Correctness gates for the Mamba optimizations. Fast; run this before trusting any
# speedup number from tests/slurm_bench.sh.
source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/

echo "=== node: $(hostname) ==="
nvidia-smi -L
pixi run python -m pytest tests/test_equivalence.py -v -s --no-header
