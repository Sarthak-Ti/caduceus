#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=dsqtl_base

source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

OUTPUT=$1
CKPT=$2

pixi run python dsqtl_onemodel.py -o "$OUTPUT" --ckpt_path "$CKPT"
# pixi run python dsqtl_onemodel.py -o base_dsqtl.npy --ckpt_path /data1/lesliec/sarthak/caduceus/outputs/2025-03-27/16-43-18-348625/checkpoints/last.ckpt