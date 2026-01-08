#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=joint_nomlm_maskonly_DE_all
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nproc
nvidia-smi

WORKERS=$((SLURM_CPUS_PER_TASK - 1))
NUM_GPUS=$(nvidia-smi -L |  wc -l)

#if want to continue, set pretrained model path and safetensors path to null but define checkpoint
pixi run srun python -m train wandb.group=joint_pretrain wandb.name=$SLURM_JOB_NAME experiment=hg38/joint_pretrain dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=$WORKERS loader.num_workers=$WORKERS model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
 trainer.devices=$NUM_GPUS \
 \
 model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
 model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
 optimizer.lr="1e-3" +train.remove_test_loader_in_eval=true \
 \
 train.task2=reg train.custom_metric=poisson_loss_mask dataset.acc_type=continuous \
 \
 dataset.data_path=/data1/lesliec/sarthak/data/DE_danwei/processed_bigwigs/DE.zarr \
 dataset.load_in=false +dataset.data_idxs=all scheduler=step trainer.accumulate_grad_batches=16 +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed \
 \
 +model.config.skip_embedding=true trainer.limit_train_batches=0.25 trainer.limit_val_batches=0.5 \
 +dataset.mask_only=true dataset.acc_mlm=0.25 dataset.mlm=0
#  +decoder.upsample=4 +encoder.downsample=4 \
#  train.ckpt="/data1/lesliec/sarthak/caduceus/outputs/2025-07-23/01-14-32-915401/checkpoints/last.ckpt" +train.pretrained_model_state_hook.load_decoder=true

# now let's set it to gpu 3 and then run it
# CUDA_VISIBLE_DEVICES=2 python -m train wandb=null experiment=hg38/joint_pretrain dataset.batch_size=1 \
#  trainer.precision=bf16 dataset.num_workers=1 loader.num_workers=1 model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
#  \
#   model=caduceus model.config.d_model=1024 model.config.n_layer=14 model.config.bidirectional=true \
#  model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
#  optimizer.lr="1e-3" +train.remove_test_loader_in_eval=true \
#  \
#  train.task2=reg train.custom_metric=poisson_loss_mask dataset.acc_type=continuous \
#  \
#  dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/dnase_chunkchrom_reprocessed.zarr \
#  dataset.load_in=false +dataset.data_idxs=all scheduler=step trainer.accumulate_grad_batches=16 trainer.limit_train_batches=0.1 trainer.limit_val_batches=0.25 \
#  \
#  +model.config.skip_embedding=true trainer.devices=1 callbacks=[base,model_every_epoch,model_every_n_steps] \
#  +dataset.mask_only=true dataset.acc_mlm=0.25 dataset.mlm=0 +decoder.upsample=4 +encoder.downsample=4 \
#  train.ckpt=/data1/lesliec/sarthak/caduceus/outputs/2025-07-23/01-14-32-915401/checkpoints/last.ckpt +train.pretrained_model_state_hook.load_decoder=true


# #if want mroe callbacks
# CUDA_VISIBLE_DEVICES=2 python -m train wandb=null experiment=hg38/joint_pretrain dataset.batch_size=1 \
#  trainer.precision=bf16 dataset.num_workers=1 loader.num_workers=1 model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
#  \
#  model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
#  model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
#  optimizer.lr="1e-3" +train.remove_test_loader_in_eval=true \
#  \
#  train.task2=reg train.custom_metric=poisson_loss_mask dataset.acc_type=continuous \
#  \
#  dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/dnase_immune.zarr \
#  dataset.load_in=false +dataset.data_idxs=all callbacks=[base,model_every_epoch,model_every_n_steps] \
#  \
#  +model.config.skip_embedding=true trainer.devices=1 \
#  +dataset.mask_only=true dataset.acc_mlm=0.25 dataset.mlm=0 \