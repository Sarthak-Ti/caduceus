#!/bin/bash

#SBATCH --partition=lesliec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=bf16_e2g_test
#SBATCH --output=jobs/%x_%j.out

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

#-- does bf16 autocast change the E2G scored delta? --
#Runs the first 50 in-context CRISPR pairs twice through the same Evals object, fp32 and bf16,
#and reports max|delta_fp32 - delta_bf16| against std(delta_fp32). Reads only; edits nothing.
#Short job: 2 forwards per element, 50 elements.

pixi run python -u test_bf16_e2g.py \
    --n_elements 50 \
    --save_npz /data1/lesliec/sarthak/data/joint_playground/e2g/bf16_vs_fp32.npz
