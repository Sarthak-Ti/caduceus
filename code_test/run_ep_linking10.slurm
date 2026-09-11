#!/bin/bash
#SBATCH --job-name=EP10_specificity
#SBATCH --partition=cpu
#SBATCH --account=lesliec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data1/lesliec/sarthak/caduceus/code_test/EP_linking10_cache/slurm-%j.log
set -euo pipefail
cd /data1/lesliec/sarthak/caduceus
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/data1/lesliec/sarthak/caduceus/code_test/EP_linking10_cache/matplotlib
export JUPYTER_RUNTIME_DIR=/data1/lesliec/sarthak/caduceus/code_test/EP_linking10_cache/jupyter_runtime
mkdir -p "$MPLCONFIGDIR" "$JUPYTER_RUNTIME_DIR"
.pixi/envs/default/bin/python -u code_test/run_ep_linking10.py "$@"
