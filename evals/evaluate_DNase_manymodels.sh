#!/bin/bash

#SBATCH --partition=gpu,lesliec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=enformer_dnase

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi
#if want to continue, set pretrained model path and safetensors path to null but define checkpoint
pixi run python eval_many_models.py
