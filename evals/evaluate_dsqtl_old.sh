#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=dsqtl_benchmark
#SBATCH --output=jobs/%x_%j.out

source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

OUTPUT=$1
CKPT=$2
MASK_SIZE=$3

pixi run python dsqtl_onemodel.py -o "$OUTPUT" --ckpt_path "$CKPT" --mask_size "$MASK_SIZE"
# pixi run python dsqtl_onemodel.py -o base_dsqtl.npy --ckpt_path /data1/lesliec/sarthak/caduceus/outputs/2025-03-27/16-43-18-348625/checkpoints/last.ckpt