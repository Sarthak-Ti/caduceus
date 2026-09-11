#!/bin/bash
#SBATCH --partition=preemptable
#SBATCH --account=preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --job-name=striped_e
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi -L; mkdir -p tests/results

echo "########## is use_mem_eff_path a pure implementation change? ##########"
pixi run python -u -m pytest tests/test_striped_equivalence.py -v -s --no-header \
    || echo "MEM_EFF EQUIVALENCE FAILED"

echo "########## upsample op optimizations ##########"
pixi run python -u -m tests.bench_upsample_opt --seqlen 262144 \
    --out tests/results/upsample_opt_262144.json || echo "UPSAMPLE OPT FAILED"
echo "=== done ==="
