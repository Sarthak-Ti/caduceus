#!/bin/bash

SCRIPT=/data1/lesliec/sarthak/caduceus/slurm_scripts/train_joint_asa_transformer.sh
MOUSE_DIR=/data1/lesliec/sarthak/data/allele_specific_mouse

B6_DATA=${MOUSE_DIR}/signal_b6_only.npz
B6_BED=${MOUSE_DIR}/sequences_b6.bed
COMBINED_DATA=${MOUSE_DIR}/signal_combined.npz
COMBINED_BED=${MOUSE_DIR}/sequences_b6_cast.bed

sbatch --begin=now+7days --job-name=mouse_b6_mlm025 "$SCRIPT" \
    dataset.data_path=${B6_DATA} \
    +dataset.sequences_bed_file=${B6_BED} \
    dataset.mlm=0.25

sbatch --begin=now+7days --job-name=mouse_combined_mlm025 "$SCRIPT" \
    dataset.data_path=${COMBINED_DATA} \
    +dataset.sequences_bed_file=${COMBINED_BED} \
    dataset.mlm=0.25

sbatch --begin=now+7days --job-name=mouse_b6_mlm100 "$SCRIPT" \
    dataset.data_path=${B6_DATA} \
    +dataset.sequences_bed_file=${B6_BED} \
    dataset.mlm=1.0

sbatch --begin=now+7days --job-name=mouse_combined_mlm100 "$SCRIPT" \
    dataset.data_path=${COMBINED_DATA} \
    +dataset.sequences_bed_file=${COMBINED_BED} \
    dataset.mlm=1.0