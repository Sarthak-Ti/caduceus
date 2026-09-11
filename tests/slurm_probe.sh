#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=mamba_probe
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Isolates whether fusing the two directions' kernels helps at all, and measures the
# conservative shared-in_proj variant that keeps mamba_inner_fn's memory behaviour.
source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi -L; mkdir -p tests/results
TAG="probe_$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')"

pixi run python -m pytest tests/test_equivalence.py -v -s --no-header -k shared_in_proj \
    || echo "shared_in_proj EQUIVALENCE FAILED"

pixi run python -m tests.bench_scan_fusion --seqlen 262144 \
    --out "tests/results/${TAG}_scanfusion.json" || echo "scan fusion probe FAILED"

for V in baseline shared_in_proj shared_in_proj_ckpt; do
  pixi run python -m tests.bench_mamba --mode backbone --variant "${V}" \
      --seqlens 131072 262144 \
      --out "tests/results/${TAG}_backbone_${V}.json" || echo "${V} FAILED"
done
echo "=== done ==="
