#!/bin/bash
#SBATCH --partition=preemptable
#SBATCH --account=preemptable
#SBATCH --qos=preemptable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --no-requeue
#SBATCH --job-name=ai_striped_gp128
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out
set -euo pipefail
cd /data1/lesliec/sarthak/caduceus
export PATH="/home/tiwaris2/.pixi/bin:$PATH"
export OMP_NUM_THREADS=4
out="$PWD/tests/results/cismamba_speedup/allelic_imbalance/striped_gp128_${SLURM_JOB_ID}"
ckpt="$PWD/outputs/cismamba_speedup_20260918/striped_gp128_screen/checkpoints/29-val_loss=0.27300.ckpt"
pixi run --frozen python -u /data1/lesliec/sarthak/data/allelic_imbalance/allelic_imbalance_onemodel.py \
 --variants /data1/lesliec/sarthak/data/allelic_imbalance/gm12878_all_positives_v2/benchmark.tsv \
 --ckpt_path "$ckpt" --data_path /data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz \
 --output-dir "$out" --mask_size 1000 --out_size 500 --pool 1 --ctt_val 6 --load_data
pixi run --frozen python tests/score_cismamba_allelic_imbalance.py "$out"
