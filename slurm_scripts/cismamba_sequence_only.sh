#!/bin/bash
#SBATCH --partition=preemptable
#SBATCH --account=preemptable
#SBATCH --qos=preemptable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=168:00:00
#SBATCH --requeue
#SBATCH --exclude=iscb017,iscb019
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out
set -euo pipefail
cd /data1/lesliec/sarthak/caduceus
export PATH="/home/tiwaris2/.pixi/bin:$PATH"
export OMP_NUM_THREADS=4
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/cismamba_seqonly_mpl_${SLURM_JOB_ID:-local}"
action=${1:?config, preflight, train, or eval required}
epochs=${2:-30}
exec pixi run --frozen python -u -m tests.cismamba_sequence_only "$action" --epochs "$epochs"
