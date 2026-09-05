#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem=200G
#SBATCH --job-name=validate_borzoi_zarr
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%x_%j.out
source ~/.bashrc
cd /data1/lesliec/sarthak/borzoi
export CUDA_VISIBLE_DEVICES=""
pixi run python /data1/lesliec/sarthak/caduceus/claude_summaries/scratch/validate_borzoi_zarr.py
