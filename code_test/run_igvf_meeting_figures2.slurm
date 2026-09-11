#!/bin/bash
# Build the IGVF_meeting_figures2 cache headlessly.
#   sbatch run_igvf_meeting_figures2.slurm            # all three stages
#   sbatch run_igvf_meeting_figures2.slurm borzoi     # one stage
# The borzoi stage is the expensive one: it reads a 339 MB target plus seven
# 339 MB prediction arrays, one arm resident at a time, so 96G is comfortable.
#SBATCH --job-name=IGVF2_figures
#SBATCH --partition=cpu
#SBATCH --account=lesliec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/data1/lesliec/sarthak/caduceus/code_test/IGVF_meeting_figures2_cache/slurm-%j.log
set -euo pipefail
cd /data1/lesliec/sarthak/caduceus
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/data1/lesliec/sarthak/caduceus/code_test/IGVF_meeting_figures2_cache/matplotlib
mkdir -p "$MPLCONFIGDIR"
.pixi/envs/default/bin/python -u code_test/igvf_meeting_figures2_pipeline.py "${1:-all}"
