#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=50:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=dsqtl_benchmark
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=0-4

# 4 tasks: IDs 0,1,2,3 if 0-3

# source ~/.bashrc
cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi


outputs=("gm12878_atac.npy" "gm12878_seq2func.npy" "gm12878_1seqmask" "gm12878_3seqmask" "gm12878_5seqmask")

ckpts=( \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-10-04/04-04-25-175957/checkpoints/last.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-09-30/15-57-16-166253/checkpoints/last.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-09-30/15-54-16-240020/checkpoints/last.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-09-30/15-54-16-242280/checkpoints/last.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-09-30/15-54-20-195512/checkpoints/last.ckpt" \
)

data_paths=( \
  "/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_ATAC_pvalue.npz" \
  "/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz" \
  "/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz" \
  "/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz" \
  "/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz" \
)

mask_sizes=(1000 524288 1000 1000 1000)

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
  # --pool 128 \
  # --out_size 196608 \
  # --load_data
#data idxs is only if trained on multiple cell types
#pool is if want to pool outputs then predict