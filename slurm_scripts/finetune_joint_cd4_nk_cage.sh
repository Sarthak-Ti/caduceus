#!/bin/bash
#SBATCH --partition=lesliec,gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=cismamba_cage
set -euo pipefail
CAGE_CONFIG=${1:?experiment.json required}
CAGE_CELL=${2:?CD4 or NK required}
CAGE_MODE=${3:-train}
case "$CAGE_MODE" in
  train|--resume) ;;
  *) echo "Third argument must be --resume or omitted" >&2; exit 2 ;;
esac
CAGE_RUN=$(cd -- "$(dirname -- "$CAGE_CONFIG")" && pwd)
CAGE_SCRIPTS=/data1/lesliec/sarthak/caduceus/slurm_scripts
CAGE_REPO=/data1/lesliec/sarthak/caduceus
CAGE_PYTHON=$CAGE_REPO/.pixi/envs/default/bin/python
export PATH="$CAGE_REPO/.pixi/envs/default/bin:$PATH"
export PYTHONPATH="$CAGE_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
# Worker sockets have a short AF_UNIX path limit; use node-local scratch.
export TMPDIR="/tmp/cage_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$TMPDIR/matplotlib" XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$CAGE_RUN/$CAGE_CELL"
cd "$CAGE_REPO"
nvidia-smi
# Fail immediately if the node cannot create the worker socket.
"$CAGE_PYTHON" -B -u - <<'PY'
from multiprocessing.connection import Listener
with Listener(family="AF_UNIX") as listener:
    print("Worker socket check passed:", listener.address, flush=True)
PY
# Resume has already passed the GPU smoke test in the original run. Restore its
# complete CAGE checkpoint directly, without fresh pretraining initialization.
if [[ "$CAGE_MODE" == "--resume" ]]; then
  exec srun --cpu-bind=none --kill-on-bad-exit=1 "$CAGE_PYTHON" -B -u \
    "$CAGE_SCRIPTS/finetune_joint_cd4_nk_cage.py" "$CAGE_CONFIG" "$CAGE_CELL" train --resume
fi
# Test the actual two-GPU train/validation path, then reinitialize from the source checkpoint.
timeout --signal=TERM --kill-after=60s 20m srun --cpu-bind=none --kill-on-bad-exit=1 "$CAGE_PYTHON" -B -u \
  "$CAGE_SCRIPTS/finetune_joint_cd4_nk_cage.py" "$CAGE_CONFIG" "$CAGE_CELL" smoke \
  > "$CAGE_RUN/$CAGE_CELL/smoke.log" 2>&1
date -u +%FT%TZ > "$CAGE_RUN/$CAGE_CELL/SMOKE_PASSED"
exec srun --cpu-bind=none --kill-on-bad-exit=1 "$CAGE_PYTHON" -B -u \
  "$CAGE_SCRIPTS/finetune_joint_cd4_nk_cage.py" "$CAGE_CONFIG" "$CAGE_CELL" train
