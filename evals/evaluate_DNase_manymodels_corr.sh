#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=enformer_dnase

# Source the bashrc file
source ~/.bashrc
#have to do lesliec partition because we need to be able to run nvidia-smi even if nothing is found!

cd /data1/lesliec/sarthak/caduceus/evals
# nvidia-smi #this is a cpu job
pixi run python eval_many_models_corr.py
cd /data1/lesliec/sarthak/data/borzoi/model_outputs/
du -sh *
