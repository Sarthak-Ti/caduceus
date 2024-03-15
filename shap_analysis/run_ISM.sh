#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# LSF directives
#BSUB -n 1
#BSUB -q gpuqueue
#BSUB -gpu "num=1"
#BSUB -R "rusage[mem=50] select[hname!=lu02 && hname!=lu08 && hname!=lu06 && hname!=lu05 && hname!=lu03 && hname!=lu04 && hname!=lu07 && hname!=lu09 && hname!=lu01]"
#BSUB -sla llSC2
#BSUB -W 20:00
#BSUB -o jobs/%J.out
#BSUB -e jobs/%J.err

# Activate your environment
source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

mamba activate /data/leslie/sarthak/environments/hyena-dna

cd /data/leslie/sarthak/hyena/hyena-dna/shap_analysis

# Run your Python training script
python ISM2.py

# ISM