#!/bin/bash
# submit_dsqtl_jobs.sh

declare -a outputs=("base_dsqtl.npy" "peaks_dsqtl.npy" "cat_dsqtl.npy")
declare -a ckpts=(
    "/data1/lesliec/sarthak/caduceus/outputs/2025-03-27/16-43-18-348625/checkpoints/last.ckpt"
    "/data1/lesliec/sarthak/caduceus/outputs/2025-03-27/16-43-39-248147/checkpoints/last.ckpt"
    "/data1/lesliec/sarthak/caduceus/outputs/2025-03-27/16-43-18-348359/checkpoints/last.ckpt"
)

for i in {0..2}; do
  sbatch evaluate_dsqtl.sh "${outputs[$i]}" "${ckpts[$i]}"
done
