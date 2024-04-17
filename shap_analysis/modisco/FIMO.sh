#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# LSF directives
#BSUB -n 1
#BSUB -R "rusage[mem=32]"
#BSUB -sla llSC2
#BSUB -W 25:00
#BSUB -o /data/leslie/sarthak/hyena/hyena-dna/jobs/%J.out
#BSUB -e /data/leslie/sarthak/hyena/hyena-dna/jobs/%J.err

# Activate your environment
source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

mamba activate /data/leslie/sarthak/environments/hyena_backup2

cd /data/leslie/sarthak/hyena/hyena-dna/shap_analysis/modisco/

# Run your Python training script
fimo --o fimo_motifs motifs_jvier_jaspar.txt sequences.fasta

# FIMO