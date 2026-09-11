#!/bin/bash
#SBATCH --partition=preemptable
#SBATCH --account=preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --job-name=striped_g
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi -L; mkdir -p tests/results
pixi run python -u -m tests.bench_convblock_opt --seqlen 262144 \
    --out tests/results/convblock_opt_262144.json || echo "FAILED"
echo "=== done ==="
