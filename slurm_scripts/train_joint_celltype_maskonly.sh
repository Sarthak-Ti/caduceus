#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=joint_cont_sepcnn_combined_celltype_nomlm_maskonly
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Same config as outputs/2025-07-18/00-23-52-538795 (joint_cont_sepcnn_combined_k562_nomlm_maskonly),
# only the npz path and wandb name differ.
# Usage: sbatch --job-name=joint_cont_sepcnn_combined_nk_nomlm_maskonly train_joint_celltype_maskonly.sh NK
#   CELL is one of: CD4T, NK, HepG2 (maps to cell_type_arrays/${CELL}_DNase.npz)
# Optional 2nd arg overrides the npz path, e.g. for ATAC:
#   sbatch --job-name=joint_cont_sepcnn_combined_gm12878atac_nomlm_maskonly train_joint_celltype_maskonly.sh GM12878 /path/to/GM12878_ATAC.npz
CELL=$1
DATA_PATH=${2:-/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/${CELL}_DNase.npz}

# Source the bashrc file
source ~/.bashrc

cd /data1/lesliec/sarthak/caduceus/
nproc
nvidia-smi

WORKERS=$((SLURM_CPUS_PER_TASK - 1))
NUM_GPUS=$(nvidia-smi -L |  wc -l)

pixi run srun --cpu-bind=none python -m train wandb.group=joint_pretrain wandb.name=$SLURM_JOB_NAME experiment=hg38/joint_pretrain dataset.batch_size=1 \
 trainer.precision=bf16 dataset.num_workers=$WORKERS loader.num_workers=$WORKERS model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
 \
 model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
 model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
 optimizer.lr="1e-3" +train.remove_test_loader_in_eval=true \
 \
 train.task2=reg train.custom_metric=poisson_loss_mask dataset.acc_type=continuous \
 \
 dataset.data_path=${DATA_PATH} \
 dataset.load_in=false +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed \
 \
 +model.config.skip_embedding=true trainer.devices=$NUM_GPUS \
 +dataset.mask_only=true dataset.acc_mlm=0.25 dataset.mlm=0
