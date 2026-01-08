#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=23:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=dsqtl_benchmark_ctt
#SBATCH --output=jobs/%x_%j.out

source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

pixi run python /data1/lesliec/sarthak/caduceus/evals/dsqtl_onemodel_ctt_legacy.py --ckpt_path /data1/lesliec/sarthak/caduceus/outputs/2025-05-15/17-49-16-937449/checkpoints/07-val_loss=0.16646.ckpt --output ctt_legacy.npy --ctt_val 6 --mask_size 1000