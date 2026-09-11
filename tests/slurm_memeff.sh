#!/bin/bash
#SBATCH --partition=preemptable
#SBATCH --account=preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --job-name=memeff_x
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi -L; mkdir -p tests/results
pixi run python -u -m tests.bench_memeff_crossover \
    --out tests/results/memeff_crossover.json || echo "FAILED"
echo "=== done ==="
