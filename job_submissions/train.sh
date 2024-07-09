#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# LSF directives
#BSUB -q gpuqueue
#BSUB -n 1
#BSUB -gpu "num=1"
#BSUB -R "A100 rusage[mem=100] span[ptile=40]"
#BSUB -sla llSC2
#BSUB -W 100:00
#BSUB -o %J.out
#BSUB -e %J.err

# Activate your environment
source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

mamba activate /data/leslie/sarthak/environments/hyena-dna

cd /data/leslie/sarthak/hyena/hyena-dna/

# Run your Python training script
python -m train wandb.group=test_wandb experiment=hg38/DNase_finetune

# End of file