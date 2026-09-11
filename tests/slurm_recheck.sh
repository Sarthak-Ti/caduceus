#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --job-name=mamba_recheck
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Re-run the variant head-to-head with the corrected timing harness (the previous runs
# emptied the allocator cache right before timing, which penalised whichever variant
# used the most memory). The bias cut *against* fused_bidir (it used more memory than
# baseline) and *for* fused_inner (it used less), so both need re-checking rather than
# arguing about the sign.
source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi -L; mkdir -p tests/results
TAG="recheck_$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')"

for V in baseline shared_in_proj fused_bidir fused_inner ckpt; do
  pixi run python -m tests.bench_mamba --mode backbone --variant "${V}" \
      --seqlens 131072 262144 --iters 5 --warmup 2 \
      --out "tests/results/${TAG}_${V}.json" || echo "${V} FAILED"
done
pixi run python -m tests.summarize_results "tests/results/${TAG}_*.json"
echo "=== done ==="
