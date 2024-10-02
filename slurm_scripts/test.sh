#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:01:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:h100:1

nvidia-smi