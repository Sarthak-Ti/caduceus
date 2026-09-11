#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=mamba_fused_inner
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Correctness + speed of BiMambaInnerFn, the properly-checkpointed fused bidirectional
# path. Correctness runs first and its result gates how the timings should be read.
source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi -L; mkdir -p tests/results
TAG="fi_$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')"

echo "########## correctness ##########"
pixi run python -m pytest tests/test_equivalence.py -v -s --no-header -k fused_inner \
    || echo "FUSED_INNER EQUIVALENCE FAILED"

echo "########## backbone head-to-head ##########"
for V in baseline fused_inner fused_inner_ckpt ckpt; do
  pixi run python -m tests.bench_mamba --mode backbone --variant "${V}" \
      --seqlens 131072 262144 524288 \
      --out "tests/results/${TAG}_backbone_${V}.json" || echo "${V} FAILED"
done

echo "########## full training step at production shape ##########"
for V in baseline fused_inner fused_inner_ckpt; do
  pixi run python -m tests.bench_mamba --mode full --variant "${V}" --seqlens 524288 \
      --out "tests/results/${TAG}_full_${V}.json" || echo "full ${V} FAILED"
done

echo "########## inference ##########"
for V in baseline fused_inner; do
  pixi run python -m tests.bench_mamba --mode infer --variant "${V}" \
      --seqlens 524288 --infer-batch-sizes 1 2 4 \
      --out "tests/results/${TAG}_infer_${V}.json" || echo "infer ${V} FAILED"
done
echo "=== done ==="
