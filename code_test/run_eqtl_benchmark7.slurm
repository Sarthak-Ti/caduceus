#!/bin/bash
#SBATCH --job-name=eQTL7_specificity
#SBATCH --partition=cpu
#SBATCH --account=lesliec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/data1/lesliec/sarthak/caduceus/code_test/eQTL_benchmark7_cache/slurm-%j.log
set -euo pipefail
cd /data1/lesliec/sarthak/caduceus
mkdir -p code_test/eQTL_benchmark7_cache
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/data1/lesliec/sarthak/caduceus/code_test/eQTL_benchmark7_cache/matplotlib
export JUPYTER_RUNTIME_DIR=/data1/lesliec/sarthak/caduceus/code_test/eQTL_benchmark7_cache/jupyter_runtime
# Completed caches are reused; use EQTL7_CACHE for a different extraction setup.
# Span reads are decompression-bound and release the GIL, so threads scale with cores.
export EQTL7_WORKERS=8
mkdir -p "$MPLCONFIGDIR" "$JUPYTER_RUNTIME_DIR"
.pixi/envs/default/bin/python -u code_test/run_eqtl_benchmark7.py "$@"
