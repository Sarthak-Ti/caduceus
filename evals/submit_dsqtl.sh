#!/bin/bash
# submit_dsqtl_jobs.sh

declare -a outputs=("nomlm_dsqtl.npy" "15mlm_dsqtl.npy" "new_acc_dsqtl.npy" "no_acc_input_dsqtl.npy")
# declare -a mask_sizes=(50 1000 0 2000 100)
declare -a mask_sizes=(1000 1000 1000 1000)
# CKPT="/data1/lesliec/sarthak/caduceus/outputs/2025-03-27/16-43-18-348625/checkpoints/last.ckpt"
declare -a CKPT=("/data1/lesliec/sarthak/caduceus/outputs/2025-04-08/11-50-37-816676/checkpoints/11-val_loss=0.27185.ckpt" \
"/data1/lesliec/sarthak/caduceus/outputs/2025-04-08/13-03-41-836697/checkpoints/11-val_loss=1.16764.ckpt" \
"/data1/lesliec/sarthak/caduceus/outputs/2025-04-08/14-47-34-262686/checkpoints/11-val_loss=1.11924.ckpt" \
"/data1/lesliec/sarthak/caduceus/outputs/2025-04-11/13-44-58-301569/checkpoints/04-val_loss=1.24072.ckpt" \
)

for i in {0..3}; do
  sbatch evaluate_dsqtl_old.sh "${outputs[$i]}" "${CKPT[$i]}" "${mask_sizes[$i]}"
done
