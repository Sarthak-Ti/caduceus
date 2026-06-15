#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=joint_gm12878_striped_maskonly_2
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nproc
nvidia-smi

WORKERS=$((SLURM_CPUS_PER_TASK - 1))
NUM_GPUS=$(nvidia-smi -L |  wc -l)

pixi run srun python -m train wandb.group=joint_pretrain wandb.name=$SLURM_JOB_NAME experiment=hg38/joint_pretrain dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=$WORKERS loader.num_workers=$WORKERS trainer.devices=$NUM_GPUS \
 \
 model=striped_hydra model.config.d_model=256 model.config.n_blocks=16 \
 model._name_=dna_embedding_striped \
 optimizer.lr="1e-3" +train.remove_test_loader_in_eval=true \
 \
 train.task2=reg train.custom_metric=poisson_loss_mask dataset.acc_type=continuous \
 \
 dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz dataset.acc_mlm=0.25 dataset.mlm=0 +dataset.mask_only=true \
 dataset.load_in=false +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed \
 \
 model.config.expand=2 +encoder.transpose=true +encoder.norm=null +encoder.activation=null \
 train.ckpt=/data1/lesliec/sarthak/caduceus/outputs/2026-04-14/17-26-24-638567/checkpoints/last.ckpt +train.pretrained_model_state_hook.load_decoder=true \

#run locally

# python -m train wandb=null experiment=hg38/joint_pretrain dataset.batch_size=1 \
#  trainer.precision=bf16 dataset.num_workers=0 loader.num_workers=0 trainer.devices=1 \
#  \
#  model=striped_hydra model.config.d_model=256 model.config.n_blocks=16 \
#  model._name_=dna_embedding_striped \
#  optimizer.lr="1e-3" +train.remove_test_loader_in_eval=true \
#  \
#  train.task2=reg train.custom_metric=poisson_loss_mask dataset.acc_type=continuous \
#  \
#  dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz dataset.acc_mlm=0.25 +dataset.mask_only=true \
#  dataset.load_in=false +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed \
#  \
#  model.config.expand=2 +encoder.transpose=true +encoder.norm=null +encoder.activation=null