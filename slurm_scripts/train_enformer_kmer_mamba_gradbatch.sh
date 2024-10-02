#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:h100:1

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nvidia-smi
#if want to continue, set pretrained model path and safetensors path to null but define checkpoint
pixi run python -m train wandb.group=enformer wandb.name=enformer_kmer_mamba_gradbatches_1_repeat experiment=hg38/enformer dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=4 dataset.rc_aug=false +dataset.mlm=false \
 model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
 model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=true \
 +decoder.conjoin_train=true +decoder.conjoin_test=false optimizer.lr="1e-3" \
 train.pretrained_model_path=null +decoder.convolutions=false dataset.max_length=196608 dataset.kmer_len=6 \
 model.config.vocab_size=15625 model.config.pad_vocab_size_multiple=1 train.pretrained_safetensors_model_path=null \
 trainer.accumulate_grad_batches=8