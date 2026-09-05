#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --job-name=extract_borzoi_cage_all
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%x_%j.out
source ~/.bashrc
cd /data1/lesliec/sarthak/caduceus
pixi run python /data1/lesliec/sarthak/caduceus/claude_summaries/scratch/extract_borzoi_cage_all.py
