#!/bin/bash
#SBATCH --partition=preemptable
#SBATCH --account=preemptable
#SBATCH --qos=preemptable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150G
#SBATCH --time=168:00:00
#SBATCH --requeue
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out
set -euo pipefail
cd /data1/lesliec/sarthak/caduceus
export PATH="/home/tiwaris2/.pixi/bin:$PATH"
export OMP_NUM_THREADS=4
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/cismamba_rna_tuning_${SLURM_JOB_ID:-local}"
exec pixi run --frozen python -u -m tests.cismamba_rna_tuning "${1:?action}" --variant "${2:?variant}"
