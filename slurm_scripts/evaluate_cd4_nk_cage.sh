#!/bin/bash
#SBATCH --partition=lesliec,gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=cismamba_cage_test
set -euo pipefail
CAGE_MANIFEST=${1:?evaluation manifest required}
CAGE_CELL=${2:?CD4 or NK required}
CAGE_REPO=/data1/lesliec/sarthak/caduceus
CAGE_PYTHON="$CAGE_REPO/.pixi/envs/default/bin/python"
export PATH="$CAGE_REPO/.pixi/envs/default/bin:$PATH"
export PYTHONPATH="$CAGE_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="/tmp/cageeval_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$TMPDIR/mpl" XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$CAGE_REPO"
nvidia-smi
exec srun --cpu-bind=none --kill-on-bad-exit=1 "$CAGE_PYTHON" -B -u \
  "$CAGE_REPO/slurm_scripts/evaluate_cd4_nk_cage.py" "$CAGE_MANIFEST" "$CAGE_CELL" --workers=4
