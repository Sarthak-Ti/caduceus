#!/bin/bash
#SBATCH --partition=preemptable
#SBATCH --account=preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --job-name=striped_f
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# What is the pre-scale rewrite actually worth on the REAL UpsampleStack, as opposed to
# on the isolated repeat+scale op? Plus a component breakdown of the top upsample layer,
# so we know whether the repeat is even a meaningful share of the stack.
source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi -L; mkdir -p tests/results
pixi run python -u -m tests.bench_upsample_opt --seqlen 262144 --real-seqlen 262144 \
    --factor 8 --out tests/results/upsample_normpatch_262144.json || echo "FAILED"
echo "=== done ==="
