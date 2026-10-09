#!/bin/bash
#SBATCH --partition=lesliec,gpu
#SBATCH --account=lesliec
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out
set -euo pipefail
cd /data1/lesliec/sarthak/caduceus
export PATH="/home/tiwaris2/.pixi/bin:$PATH"
export OMP_NUM_THREADS=4
export MPLBACKEND=Agg
label=${1:?Reference label required}
checkpoint=${2:?Checkpoint required}
result_dir="$PWD/tests/results/cismamba_tuning_20260928/caqtl/$label"
attempt="$result_dir/attempt_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}"
mkdir -p "$attempt"
pixi run --frozen python -u evals/caqtl_onemodel_varisused.py \
  --output "$attempt/predictions.npy" --ckpt_path "$checkpoint" \
  --data_path /data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz \
  --mask_size 1000 --out_size 500 --pool 1 --ctt_val 6 --load_data
# Use the reviewed immutable snapshot that reproduced the original reference
# metrics, avoiding dependence on later edits to the user's live notebook.
pixi run --frozen python -u tests/score_cismamba_caqtl.py \
  --manifest tests/results/cismamba_old_dnase_20260924/caqtl/scoring_manifest_20260928.json \
  --model "$label" --prediction "$attempt/predictions.npy" --output-dir "$result_dir"
