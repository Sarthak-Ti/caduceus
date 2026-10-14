#!/bin/bash
#SBATCH --partition=lesliec,gpu
#SBATCH --account=lesliec
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=95:00:00
#SBATCH --requeue
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out
set -euo pipefail
cd /data1/lesliec/sarthak/caduceus
export PATH="/home/tiwaris2/.pixi/bin:$PATH"
export OMP_NUM_THREADS=4
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/cismamba_dstate1_mpl_${SLURM_JOB_ID:-local}"
action=${1:?config, preflight, or train required}
variant=${2:?variant required}
epochs=${3:-30}
# d_state 1 arms need Triton >= 2.2 (2.1 aborts in the SSD scan): run them in the
# isolated Triton 3.0 venv layered over this pixi env. ms64_d256 keeps the default env.
case "$variant" in
  ms1_d256|ms1_d512|striped_s1)
    T=/data1/lesliec/sarthak/triton_upgrade_test
    export CISMAMBA_PYTHON=$T/venv_triton_3.0.0/bin/python
    export TRITON_CACHE_DIR=$T/triton_cache/triton_3.0.0 ;;
esac
exec pixi run --frozen "${CISMAMBA_PYTHON:-python}" -u -m tests.cismamba_dstate1 "$action" --variant "$variant" --epochs "$epochs"
