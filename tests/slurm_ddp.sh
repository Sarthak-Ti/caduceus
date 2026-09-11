#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --job-name=mamba_ddp
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

source ~/.bashrc
set -o pipefail
cd /data1/lesliec/sarthak/caduceus/
echo "=== node: $(hostname) ==="; nvidia-smi -L
pixi run torchrun --nproc_per_node=2 --master_port=29517 tests/test_ddp_checkpointing.py \
    || echo "DDP CHECKPOINTING TEST FAILED"
echo "=== done ==="
