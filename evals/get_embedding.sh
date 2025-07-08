#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=get_embedding
#SBATCH --output=jobs/%j-%x.out

# source ~/.bashrc
cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

pixi run python get_embedding.py