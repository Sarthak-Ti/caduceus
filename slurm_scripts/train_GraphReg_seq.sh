#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=GR_1M

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nvidia-smi
#if want to continue, set pretrained model path and safetensors path to null but define checkpoint
pixi run python -m train wandb.group=graphreg wandb.name=$SLURM_JOB_NAME experiment=hg38/GraphReg dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=$SLURM_CPUS_PER_TASK +dataset.mlm=false \
 model.config.vocab_size=12 model.config.pad_vocab_size_multiple=1 \
 \
 model=caduceus model.config.d_model=128 model.config.n_layer=8 model.config.bidirectional=true \
 model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=true \
 +decoder.conjoin_train=true +decoder.conjoin_test=false optimizer.lr="1e-3" \
 \
 train.pretrained_model_path=null +decoder.convolutions=false dataset.max_length=1000000 \
 trainer.accumulate_grad_batches=8 