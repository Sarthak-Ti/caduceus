#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# LSF directives
#BSUB -q gpuqueue
#BSUB -n 1
#BSUB -gpu "num=1:j_exclusive=yes:gmodel=UnknownNVIDIAA10080GBPCIe"
#BSUB -R "rusage[mem=100]"
#BSUB -sla llSC2
#BSUB -W 3:00
#BSUB -o /data/leslie/sarthak/hyena/hyena-dna/jobs/%J.out
#BSUB -e /data/leslie/sarthak/hyena/hyena-dna/jobs/%J.err

# Activate your environment
source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

mamba activate /data/leslie/sarthak/environments/mambadna

cd /data/leslie/sarthak/caduceus/evals/

python <<EOF
import evals_utils_enformer as e
mamba = e.Evals('/data/leslie/sarthak/caduceus/outputs/2024-08-06/16-29-46-520749/checkpoints/05-val_loss=0.61190.ckpt')
allout = mamba.evaluate(2)
import numpy as np
np.save('/data/leslie/sarthak/data/enformer/data/model_out/caduceus_kmer.npy', allout)
EOF