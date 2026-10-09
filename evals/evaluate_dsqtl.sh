#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=dsqtl_benchmark
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=3

# Submit --array=4-5 for the GM12878 ATAC and HepG2 DNase best-loss models.

# source ~/.bashrc
cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi


outputs=(
  "phastcons_redo.npy"
  "arsenal_redo.npy"
  "UNUSED_slot2.npy"
  "nomlm_maskonly_gm12878trained_hepg2eval.npy"
  "nomlm_maskonly_gm12878atac_bestloss_dsqtl.npy"
  "nomlm_maskonly_hepg2_bestloss_dsqtl.npy"
)

ckpts=( \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-12-01/14-46-22-759501/checkpoints/last.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-04-28/20-15-27-263513/checkpoints/last.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-04-30/10-45-54-347624/checkpoints/last.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-04-28/15-26-30-700432/checkpoints/14-val_loss=0.27260.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-09-24/09-53-54-581722/checkpoints/11-val_loss=0.05880.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-09-23/11-45-50-395363/checkpoints/15-val_loss=0.22884.ckpt" \
)

data_paths=( \
  "/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz" \
  "/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz" \
  "/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz" \
  "/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/HepG2_DNase.npz" \
  "/data1/lesliec/sarthak/data/alphagenome/processed_bigwigs/gm12878_atac_unmap_only.npz" \
  "/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/HepG2_DNase.npz" \
)

mask_sizes=(1000 1000 1000 1000 1000 1000)

#–– pick the right one based on SLURM_ARRAY_TASK_ID ––
i=$SLURM_ARRAY_TASK_ID
OUTPUT=${outputs[$i]}
CKPT=${ckpts[$i]}
MASK_SIZE=${mask_sizes[$i]}
data_path=${data_paths[$i]}

echo "Running task $i with output $OUTPUT, ckpt $CKPT, data path $data_path, mask size $MASK_SIZE"
#–– run the ask ––
pixi run python -u dsqtl_onemodel.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --mask_size "$MASK_SIZE" \
  --data_path "$data_path" \
  --load_data \
  # --ctt_val 4 \
  # --encoder_numcelltypes 7 \
  # --pool 128 \
  # --out_size 196608 \
  # --load_data
#data idxs is only if trained on multiple cell types
#pool is if want to pool outputs then predict
