#!/bin/bash
#SBATCH --partition=gpu,lesliec
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --job-name=mamba_smoke
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Cheap end-to-end exercise of every benchmark code path on any available GPU, so
# harness bugs surface here instead of inside a 2-hour A100 job.
source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi -L
mkdir -p tests/results

for V in baseline fused_bidir ckpt fused_bidir_ckpt; do
  echo "########## smoke: ${V} ##########"
  pixi run python -m tests.bench_mamba --mode all --variant "${V}" \
      --seqlens 8192 16384 --micro-seqlen 8192 --iters 2 --warmup 1 --n-layer 4 \
      --infer-batch-sizes 1 2 \
      --out "tests/results/smoke_${V}.json" || echo "SMOKE ${V} FAILED"
done

echo "########## smoke: profiler ##########"
pixi run python -m tests.profile_mamba --target backbone --seqlen 8192 --n-layer 4 \
    --iters 2 --warmup 1 --row-limit 15 || echo "SMOKE profiler FAILED"
pixi run python -m tests.profile_mamba --target full --seqlen 8192 --n-layer 4 \
    --iters 2 --warmup 1 --row-limit 15 || echo "SMOKE profiler-full FAILED"

echo "########## smoke: summarizer ##########"
pixi run python -m tests.summarize_results "tests/results/smoke_*.json" || echo "SMOKE summarize FAILED"
echo "=== done ==="
