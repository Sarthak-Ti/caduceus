#!/bin/bash
#SBATCH --partition=gpu,lesliec
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --job-name=mamba_decide
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Decisive test for fused_inner on the production GPU.
#
# With the corrected harness, fused_inner is 6-8% FASTER than baseline on H200
# (HBM3e, ~4.8 TB/s) but was 37% SLOWER on L40S (GDDR6, ~864 GB/s). The fused path
# trades fused-kernel work for memory traffic, so the verdict looks bandwidth-dependent.
# A100 80GB PCIe (~1935 GB/s) sits between them and is what production runs on.
source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi -L; mkdir -p tests/results

for V in baseline fused_inner shared_in_proj; do
  pixi run python -m tests.bench_mamba --mode backbone --variant "${V}" \
      --seqlens 131072 262144 524288 --iters 5 --warmup 2 \
      --out "tests/results/a100d_backbone_${V}.json" || echo "${V} backbone FAILED"
done

for V in baseline fused_inner; do
  pixi run python -m tests.bench_mamba --mode full --variant "${V}" --seqlens 524288 \
      --iters 5 --warmup 2 --out "tests/results/a100d_full_${V}.json" || echo "full ${V} FAILED"
  pixi run python -m tests.bench_mamba --mode infer --variant "${V}" --seqlens 524288 \
      --infer-batch-sizes 1 2 4 --iters 5 --warmup 2 \
      --out "tests/results/a100d_infer_${V}.json" || echo "infer ${V} FAILED"
done

pixi run python -m tests.summarize_results "tests/results/a100d_*.json"
echo "=== done ==="
