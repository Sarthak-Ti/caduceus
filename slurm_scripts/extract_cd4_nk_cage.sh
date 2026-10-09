#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --job-name=extract_cd4_nk_cage
set -euo pipefail
CAGE_CONFIG=${1:?experiment.json required}
CAGE_SCRIPTS=/data1/lesliec/sarthak/caduceus/slurm_scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec /data1/lesliec/sarthak/caduceus/.pixi/envs/default/bin/python -B -u \
  "$CAGE_SCRIPTS/extract_cd4_nk_cage.py" "$CAGE_CONFIG" --workers=4
