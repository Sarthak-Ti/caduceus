#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# LSF directives
#BSUB -n 1
#BSUB -q gpuqueue
#BSUB -gpu "num=1"
#BSUB -R "A100 rusage[mem=50]"
#BSUB -sla llSC2
#BSUB -W 68:00
#BSUB -o jobs/%J.out
#BSUB -e jobs/%J.err

# Activate your environment
source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

mamba activate /data/leslie/sarthak/environments/hyena-dna

cd /data/leslie/sarthak/hyena/hyena-dna/shap_analysis

# Run your Python training script
python ISM4.py

# ISM