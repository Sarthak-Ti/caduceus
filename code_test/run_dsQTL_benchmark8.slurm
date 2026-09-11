#!/bin/bash
#SBATCH --job-name=dsQTL8_specificity
#SBATCH --partition=cpu
#SBATCH --account=lesliec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/data1/lesliec/sarthak/caduceus/code_test/dsQTL_benchmark8_cache/slurm-%j.log
set -euo pipefail
cd /data1/lesliec/sarthak/caduceus
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/data1/lesliec/sarthak/caduceus/code_test/dsQTL_benchmark8_cache/matplotlib
export JUPYTER_RUNTIME_DIR=/data1/lesliec/sarthak/caduceus/code_test/dsQTL_benchmark8_cache/jupyter_runtime
export DSQTL8_BUILD=1 DSQTL8_WORKERS=4
mkdir -p "$MPLCONFIGDIR" "$JUPYTER_RUNTIME_DIR"
.pixi/envs/default/bin/python -u code_test/run_dsQTL_benchmark8.py
