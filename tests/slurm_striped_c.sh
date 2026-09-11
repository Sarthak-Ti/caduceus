#!/bin/bash
#SBATCH --partition=preemptable
#SBATCH --account=preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --job-name=striped_c
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi -L; mkdir -p tests/results

pixi run python -u -m tests.bench_hydra_isolate --seqlen 262144 \
    --out tests/results/hydra_isolate_262144.json || echo "ISOLATE 262144 FAILED"
pixi run python -u -m tests.bench_hydra_isolate --seqlen 65536 \
    --out tests/results/hydra_isolate_65536.json || echo "ISOLATE 65536 FAILED"
echo "=== done ==="
