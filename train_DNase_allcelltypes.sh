#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# LSF directives
#BSUB -q gpuqueue
#BSUB -n 2
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "A100 rusage[mem=100]"
#BSUB -sla llSC2
#BSUB -W 48:00
#BSUB -o jobs/%J.out
#BSUB -e jobs/%J.err

# Activate your environment
source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

mamba activate /data/leslie/sarthak/environments/hyena-dna

cd /data/leslie/sarthak/hyena/hyena-dna/

# Run your Python training script
python -m train wandb.group=DNase_class_multitask experiment=hg38/DNase_allcelltypes_classification_finetune

# DNase allcelltypes train