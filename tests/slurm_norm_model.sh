#!/bin/bash
#SBATCH --partition=preemptable
#SBATCH --account=preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:35:00
#SBATCH --job-name=norm_model
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi -L; mkdir -p tests/results
pixi run python -u -m tests.bench_norm_patch_model --seqlens 262144 524288 \
    --out tests/results/norm_patch_model.json || echo "FAILED"
echo "=== done ==="
