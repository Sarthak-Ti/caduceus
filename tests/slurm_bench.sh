#!/bin/bash
#SBATCH --partition=gpu,lesliec
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=mamba_bench
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Benchmark + profile the pure Mamba (Caduceus) architecture.
# Usage: sbatch tests/slurm_bench.sh [TAG] [EXTRA_ARGS...]
#   TAG defaults to "baseline" and names the output json/log files.

source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/

TAG="${1:-baseline}"
shift || true

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-none} ==="
nvidia-smi

mkdir -p tests/results

echo
echo "############ BENCHMARK SWEEP (${TAG}) ############"
pixi run python -m tests.bench_mamba \
    --mode all \
    --variant "${TAG}" \
    --out "tests/results/bench_${TAG}.json" "$@"

echo
echo "############ PROFILE: backbone ############"
pixi run python -m tests.profile_mamba \
    --target backbone --seqlen 131072 --check-launch-bound \
    --trace "tests/results/trace_backbone_${TAG}.json"

echo
echo "############ PROFILE: full step ############"
pixi run python -m tests.profile_mamba \
    --target full --seqlen 131072

echo "=== done ==="
