#!/bin/bash
#SBATCH --partition=lesliec,gpu
#SBATCH --account=lesliec
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --job-name=e2g_poisson_profile
#SBATCH --output=/data1/lesliec/sarthak/caduceus/evals/jobs/%j-e2g_poisson_profile.out
set -euo pipefail
# Usage: sbatch [Slurm options] evals/evaluate_e2g_tss_profile_readout.sh CHECKPOINT OUTPUT_DIR [evaluator options]
if (( $# < 2 )); then
  printf 'Usage: %s CHECKPOINT OUTPUT_DIR [evaluator options]\n' "$0" >&2
  exit 2
fi
E2G_CHECKPOINT=$(realpath -e "$1")
E2G_OUTPUT_DIR=$(realpath -m "$2")
shift 2
cd /data1/lesliec/sarthak/caduceus
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/e2g_poisson_profile_mpl_${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec .pixi/envs/default/bin/python -u evals/e2g_tss_profile_readout.py \
  --ckpt-path "$E2G_CHECKPOINT" --output-dir "$E2G_OUTPUT_DIR" "$@"
