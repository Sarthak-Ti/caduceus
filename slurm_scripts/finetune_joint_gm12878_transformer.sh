#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=3
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=finetune_gm12878_transformer
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nproc
nvidia-smi

WORKERS=$((SLURM_CPUS_PER_TASK - 1))
NUM_GPUS=$(nvidia-smi -L |  wc -l)

pixi run srun python -m train wandb.group=joint_pretrain wandb.name=$SLURM_JOB_NAME experiment=hg38/joint_finetune dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=$WORKERS loader.num_workers=$WORKERS trainer.devices=$NUM_GPUS \
 \
 model=striped_hydra model._name_=dna_embedding_striped \
 model.config.d_model=256 model.config.mode=transformer_only model.config.n_blocks=16 \
 model.config.residual_in_fp32=true \
 +model.config.d_in=128 +model.config.global_pooling=128 \
 +model.config.dropout=0.1 +model.config.attention_dropout=0.1 +model.config.sampling_checkpoint=true \
 optimizer.lr="1e-4" +train.remove_test_loader_in_eval=true \
 \
 dataset.acc_type=continuous \
 \
 dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz \
 dataset.load_in=true +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed \
 \
 +encoder.transpose=true +encoder.norm=null +encoder.activation=null \
 \
 +decoder.conjoin_train=false +decoder.conjoin_test=false +decoder.convolutions=true +encoder.d_input2=2 \
 +decoder.d_model=128 +decoder.d_output=1 trainer.accumulate_grad_batches=16 \
 +dataset.additional_data=/data1/lesliec/sarthak/data/enformer/data/GM12878CAGE.npz task.loss._name_=poisson_loss_nan \
 \
 train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2026-04-30/10-45-54-347624/checkpoints/98-val_loss\=0.27203.ckpt"

#run locally

# python -m train wandb=null experiment=hg38/joint_finetune dataset.batch_size=2 \
#  trainer.precision=bf16 dataset.num_workers=0 loader.num_workers=0 trainer.devices=1 \
#  \
#  model=striped_hydra model._name_=dna_embedding_striped \
#  model.config.d_model=256 model.config.mode=transformer_only model.config.n_blocks=16 \
#  model.config.residual_in_fp32=true \
#  +model.config.d_in=128 +model.config.global_pooling=128 \
#  +model.config.dropout=0.1 +model.config.attention_dropout=0.1 +model.config.sampling_checkpoint=true \
#  task.loss._name_=poisson_loss_nan \
#  optimizer.lr="1e-3" +train.remove_test_loader_in_eval=true \
#  \
#  dataset.acc_type=continuous \
#  \
#  dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz \
#  dataset.load_in=true +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed \
#  \
#  +encoder.transpose=true +encoder.norm=null +encoder.activation=null \
#  \
#  +decoder.conjoin_train=false +decoder.conjoin_test=false +decoder.convolutions=false \
#  +decoder.d_model=128 +decoder.d_output=1 \
#  \
#  +callbacks=model_every_n_steps callbacks.model_checkpoint_every_n_steps.every_n_train_steps=1000 trainer.accumulate_grad_batches=16 \
#  train.pretrained_model_path="..."