#!/bin/bash
#SBATCH --partition=lesliec,gpu
#SBATCH --account=lesliec
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --job-name=cd4_nk_wide256
set -euo pipefail
WIDE_REPO=/data1/lesliec/sarthak/caduceus
WIDE_MANIFEST=${1:?manifest.json required}
WIDE_CELL=${2:?CD4 or NK required}
WIDE_STAGE=${3:?pretrain or finetune required}
export PATH="$WIDE_REPO/.pixi/envs/default/bin:$PATH"
export PYTHONPATH="$WIDE_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
# Keep multiprocessing's AF_UNIX socket paths short.
export TMPDIR="/tmp/w256_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$TMPDIR/mpl" XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$WIDE_REPO"
nvidia-smi
"$WIDE_REPO/.pixi/envs/default/bin/python" -B - <<'PY'
from multiprocessing.connection import Listener
with Listener(family='AF_UNIX') as listener:
    print('Worker socket check passed:', listener.address, flush=True)
PY
exec "$WIDE_REPO/.pixi/envs/default/bin/python" -B -u \
  "$WIDE_REPO/slurm_scripts/cismamba_cd4_nk_wide256.py" stage \
  --manifest "$WIDE_MANIFEST" --cell "$WIDE_CELL" --stage "$WIDE_STAGE"
