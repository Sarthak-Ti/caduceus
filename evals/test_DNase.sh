#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# LSF directives
#BSUB -q gpuqueue
#BSUB -n 1
#BSUB -gpu "num=1"
#BSUB -R "rusage[mem=32]"
#BSUB -sla llSC2
#BSUB -W 25:00
#BSUB -o /data/leslie/sarthak/hyena/hyena-dna/jobs/%J.out
#BSUB -e /data/leslie/sarthak/hyena/hyena-dna/jobs/%J.err

# Activate your environment
source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

mamba activate /data/leslie/sarthak/environments/hyena-dna

cd /data/leslie/sarthak/hyena/hyena-dna/evals/

# Run your Python training script
python eval_classification.py

# DNase test