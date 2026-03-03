#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=k562_tss_bulk_finetune
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nproc
nvidia-smi

WORKERS=$((SLURM_CPUS_PER_TASK - 1))
NUM_GPUS=$(nvidia-smi -L | wc -l)

pixi run srun python -m train wandb.group=tss_finetune wandb.name=$SLURM_JOB_NAME experiment=hg38/joint_finetune dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=$WORKERS loader.num_workers=$WORKERS model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
 \
 model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
 model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
 optimizer.lr="1e-4" +train.remove_test_loader_in_eval=true \
 \
 dataset._name_=TSSLoader \
 dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/K562_DNase.npz \
 dataset.load_in=false \
 dataset.shift_sequences=2000 \
 +dataset.rc_strand=true \
 +dataset.tss_json_file=/data1/lesliec/sarthak/data/DE_danwei/k562_bulk_rna_info.json \
 dataset.acc_type=continuous \
 \
 +model.config.skip_embedding=true trainer.devices=$NUM_GPUS \
 \
 task._name_=joint_tss \
 task.loss._name_=mse_tss \
 task.metrics=[mse_tss] \
 \
 decoder._name_=tss +decoder.d_output=1 +decoder.hidden_dim=128 \
 train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2025-07-18/00-23-52-538795/checkpoints/last.ckpt"


# CUDA_VISIBLE_DEVICES=0 python -m train wandb=null experiment=hg38/joint_finetune dataset.batch_size=1 \
#  trainer.precision=bf16 dataset.num_workers=1 loader.num_workers=1 model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
#  \
#  model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
#  model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
#  optimizer.lr="1e-4" +train.remove_test_loader_in_eval=true \
#  \
#  dataset._name_=TSSLoader \
#  dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/K562_DNase.npz \
#  dataset.load_in=false \
#  dataset.shift_sequences=2000 \
#  +dataset.rc_strand=true \
#  +dataset.tss_json_file=/data1/lesliec/sarthak/data/DE_danwei/k562_bulk_rna_info.json \
#  dataset.acc_type=continuous \
#  \
#  +model.config.skip_embedding=true trainer.devices=1 \
#  \
#  task._name_=joint_tss \
#  task.loss._name_=mse_tss \
#  task.metrics=[mse_tss] \
#  \
#  decoder._name_=tss +decoder.d_output=1 +decoder.hidden_dim=128 \
#  train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2025-07-18/00-23-52-538795/checkpoints/last.ckpt"
