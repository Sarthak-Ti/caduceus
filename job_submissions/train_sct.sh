#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# LSF directives
#BSUB -q gpuqueue
#BSUB -n 4
#BSUB -gpu "num=1:j_exclusive=yes:gmodel=UnknownNVIDIAA10080GBPCIe"
#BSUB -R "A100 rusage[mem=100]"
#BSUB -sla llSC2
#BSUB -W 20:00
#BSUB -o jobs/%J.out
#BSUB -e jobs/%J.err

# Activate your environment
source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

mamba activate /data/leslie/sarthak/environments/hyena-dna

cd /data/leslie/sarthak/hyena/hyena-dna/


#the goal is to train individual cell types!
for i in {155..160}
do
    python -m train wandb.group=DNase_sct wandb.name=celltype_$i experiment=hg38/DNase_ctst_singlecelltype dataset.single_cell_type=$i
done

i=80
python -m train wandb.group=DNase_sct wandb.name=celltype_${i}_redo experiment=hg38/DNase_ctst_singlecelltype dataset.single_cell_type=$i
i=154
python -m train wandb.group=DNase_sct wandb.name=celltype_${i}_redo experiment=hg38/DNase_ctst_singlecelltype dataset.single_cell_type=$i

#final models