#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=mamba_attrib
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Bottleneck attribution on whatever GPU is free. The A100 job (tests/slurm_bench.sh)
# owns the headline latency/throughput numbers; this one exists to answer "where does
# the time go", which is largely a property of the architecture rather than the GPU.
source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi; mkdir -p tests/results
TAG="attrib_$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')"

echo "########## per-op attribution + backbone sweep (baseline) ##########"
pixi run python -m tests.bench_mamba --mode micro --micro-seqlen 262144 \
    --out "tests/results/${TAG}_micro.json" || echo "micro FAILED"

pixi run python -m tests.bench_mamba --mode backbone --variant baseline \
    --seqlens 65536 131072 262144 524288 \
    --out "tests/results/${TAG}_backbone_baseline.json" || echo "backbone FAILED"

echo "########## kernel profile ##########"
pixi run python -m tests.profile_mamba --target backbone --seqlen 262144 \
    --check-launch-bound --row-limit 30 || echo "profile FAILED"
pixi run python -m tests.profile_mamba --target full --seqlen 262144 \
    --row-limit 20 || echo "profile-full FAILED"

echo "########## variant head-to-head ##########"
for V in fused_bidir ckpt fused_bidir_ckpt; do
  pixi run python -m tests.bench_mamba --mode backbone --variant "${V}" \
      --seqlens 65536 131072 262144 524288 \
      --out "tests/results/${TAG}_backbone_${V}.json" || echo "${V} FAILED"
done

echo "########## batch scaling at production length ##########"
for V in ckpt fused_bidir_ckpt; do
  for B in 1 2 4; do
    pixi run python -m tests.bench_mamba --mode full --variant "${V}" \
        --seqlens 524288 --batch-size ${B} \
        --out "tests/results/${TAG}_full_${V}_b${B}.json" || echo "${V} b${B} FAILED"
  done
done
pixi run python -m tests.bench_mamba --mode full --variant baseline --seqlens 524288 \
    --out "tests/results/${TAG}_full_baseline_b1.json" || echo "baseline full FAILED"
echo "=== done ==="
