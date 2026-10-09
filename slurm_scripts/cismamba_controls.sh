#!/bin/bash
#SBATCH --partition=lesliec
#SBATCH --account=lesliec
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=168:00:00
#SBATCH --requeue
#SBATCH --job-name=cismamba_controls
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out
set -euo pipefail
cd /data1/lesliec/sarthak/caduceus
export PATH="/home/tiwaris2/.pixi/bin:$PATH"
export OMP_NUM_THREADS=4
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/cismamba_controls_mpl_${SLURM_JOB_ID:-local}"
action=${1:?preflight, train, or eval required}
variant=${2:?variant required}
epochs=${3:-30}
case "$action" in
  preflight)
    pixi run --frozen python -u -m tests.cismamba_controls benchmark --variant "$variant"
    pixi run --frozen python -u -m tests.cismamba_controls smoke --variant "$variant"
    ;;
  train|eval)
    exec pixi run --frozen python -u -m tests.cismamba_controls "$action" --variant "$variant" --epochs "$epochs"
    ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
