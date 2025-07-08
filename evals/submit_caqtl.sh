#!/bin/bash

#basic script, it creates zarr files then submits slurm jobs to do caQTL of the models

#first get checkpoints outputs and mask sizes
outputs=("nomlm_maskonly" "nomlm_maskonly_0" "base_model" "immune_nob_maskonly_nomlm")

ckpts=( \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-04-28/15-26-30-700432/checkpoints/14-val_loss=0.27260.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-04-28/15-26-30-700432/checkpoints/14-val_loss=0.27260.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-03-27/16-43-18-348625/checkpoints/15-val_loss=0.00000.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2025-04-28/15-39-56-643580/checkpoints/07-val_loss=0.14701.ckpt" \
)

mask_sizes=(1000 0 1000 1000)

TOTAL=95065

cd /data1/lesliec/sarthak/caduceus/evals
for i in "${!outputs[@]}"; do
  TEMPOUTPUT=${outputs[$i]}
  CKPT=${ckpts[$i]}
  MASK_SIZE=${mask_sizes[$i]}
  OUTPUT="/data1/lesliec/sarthak/data/joint_playground/caQTL/${TEMPOUTPUT}.zarr"

  #first we create the zarr file
  pixi run python create_zarr.py --samples $TOTAL --zarr_file $OUTPUT

  echo "Running task $i with output $OUTPUT, ckpt $CKPT, mask size $MASK_SIZE"
  #–– run the task ––
  sbatch --export=TOTAL="$TOTAL",ZARR_OUT="$OUTPUT",MASK_SIZE="$MASK_SIZE",CKPT="$CKPT" \
    evaluate_caqtl.sh
done