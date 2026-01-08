#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=dsqtl_benchmark
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-1

# 4 tasks: IDs 0,1,2,3 if 0-3

# source ~/.bashrc
cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi


outputs=("eqtlpip_cont.npy" "eqtlscaled_binary.npy")

ckpts=( \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-12-06/22-01-02-304317/checkpoints/last.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-12-06/22-05-01-508538/checkpoints/last.ckpt" \
)

data_paths=( \
  "/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz" \
  "/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz" \
)

mask_sizes=(1000 1000 1000 1000)

#–– pick the right one based on SLURM_ARRAY_TASK_ID ––
i=$SLURM_ARRAY_TASK_ID
OUTPUT=${outputs[$i]}
CKPT=${ckpts[$i]}
MASK_SIZE=${mask_sizes[$i]}
data_path=${data_paths[$i]}

echo "Running task $i with output $OUTPUT, ckpt $CKPT, mask size $MASK_SIZE"
#–– run the ask ––
pixi run python dsqtl_onemodel.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --mask_size "$MASK_SIZE" \
  --data_path "$data_path" \
  --load_data
  # --encoder_numcelltypes 7 \
  # --ctt_val 6 \
  # --pool 128 \
  # --out_size 196608 \
  # --load_data
#data idxs is only if trained on multiple cell types
#pool is if want to pool outputs then predict