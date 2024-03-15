#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# LSF directives
#BSUB -n 2
#BSUB -R "rusage[mem=500]"
#BSUB -sla llSC2
#BSUB -W 45:00
#BSUB -o jobs/%J.out
#BSUB -e jobs/%J.err

# Activate your environment
source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

mamba activate /data/leslie/sarthak/environments/hyena-dna

cd /data/leslie/sarthak/hyena/hyena-dna/shap_analysis

# Run your Python training script
python run_1_ccre.py

# one ccre