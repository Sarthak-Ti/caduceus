#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# LSF directives
#BSUB -q gpuqueue
#BSUB -n 4
#BSUB -gpu "num=1:j_exclusive=yes:gmodel=UnknownNVIDIAA10080GBPCIe"
#BSUB -R "A100 rusage[mem=30]"
#BSUB -sla llSC2
#BSUB -W 168:00
#BSUB -o jobs/%J.out
#BSUB -e jobs/%J.err

# Activate your environment
source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

mamba activate /data/leslie/sarthak/environments/hyena_backup2

cd /data/leslie/sarthak/hyena/hyena-dna/

python -m train wandb.group=enformer wandb.name=enformer_cnn-encoder_linear-out_nodownsample_fulldata_1 experiment=hg38/enformer dataset.batch_size=2 \
 trainer.precision=bf16 dataset.num_workers=4 +decoder.convolutions=false dataset.max_length=196608 dataset.rc_aug=false \
 +encoder._name_=enformer +encoder.pool_type=none +encoder.conv_tower=true train.pretrained_model_path=null
#model layer l max is fine as it shoudl just be whatever the dataset size is as we do no downsample with the cnn!!!
#enformer cnn no pool