#!/bin/bash

# Source the bashrc file
source ~/.bashrc

# LSF directives
#BSUB -q gpuqueue
#BSUB -n 1
#BSUB -gpu "num=1:j_exclusive=yes:gmodel=UnknownNVIDIAA10080GBPCIe"
#BSUB -R "A100 rusage[mem=100]"
#BSUB -sla llSC2
#BSUB -W 40:00
#BSUB -o jobs/%J.out
#BSUB -e jobs/%J.err

# Activate your environment
source ~/mambaforge/etc/profile.d/mamba.sh  # Adjust this line to your environment activation command

mamba activate /data/leslie/sarthak/environments/hyena_backup2

cd /data/leslie/sarthak/hyena/hyena-dna/

# WAIT_FILE="/data/leslie/sarthak/data/chrombpnet_test/bias_data_fixed/train/nonpeak_seqs.npy"
# wait_for_file() {
#   while [ ! -f "$WAIT_FILE" ]; do
#     echo "Waiting for $WAIT_FILE to exist..."
#     sleep 600  # Sleep for 60 seconds before checking again
#   done
# }

# Call the function to wait for the file
# wait_for_file

python -m train wandb.group=Profile_1000_800_biasmodel wandb.name=biasmodel_3 experiment=hg38/profile_nobias dataset.batch_size=2048 \
 trainer.precision=bf16 dataset.train_bias=true dataset.dataset_path=/data/leslie/sarthak/data/chrombpnet_test/bias_data_fixed/ \
 train.count_weight=4.2 \
 train.pretrained_model_path=/data/leslie/sarthak/hyena/hyena-dna/hyenadna-tiny-1k-seqlen/weights.ckpt
    # task.bias_model=/data/leslie/sarthak/data/chrombpnet_test/chrombpnet_model_1000/models/bias_model_scaled.h5
#these modifications of dataset and trainer are so will run on a100 efficiently, but noneed to adjust the yaml when debugging on another gpu
#without bias