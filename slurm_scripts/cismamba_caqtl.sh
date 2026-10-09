#!/bin/bash
#SBATCH --partition=preemptable
#SBATCH --account=preemptable
#SBATCH --qos=preemptable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --job-name=cs_caqtl
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out
set -euo pipefail
cd /data1/lesliec/sarthak/caduceus
export PATH="/home/tiwaris2/.pixi/bin:$PATH"
export OMP_NUM_THREADS=4
export MPLBACKEND=Agg
variant=${1:?Model name required}
checkpoint=${2:?Checkpoint required}
result_dir=${3:?Result directory required}
# Preserve interrupted artifacts, and use a fresh directory after preemption.
attempt="$result_dir/attempt_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}"
mkdir -p "$attempt"
pixi run --frozen python -u evals/caqtl_onemodel_varisused.py \
  --output "$attempt/predictions.npy" --ckpt_path "$checkpoint" \
  --data_path /data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz \
  --mask_size 1000 --out_size 500 --pool 1 --ctt_val 6 --load_data
pixi run --frozen python -u tests/score_cismamba_caqtl.py \
  --manifest tests/results/cismamba_caqtl_20260923/manifest.json \
  --model "$variant" --prediction "$attempt/predictions.npy" --output-dir "$result_dir"
