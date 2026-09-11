#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=k562_tss_sc_decima_scratch
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# From-scratch + ATAC arm of the K562 2x2 (pretraining x accessibility-input).
#
#            | ATAC input                        | ATAC masked
#   ---------+-----------------------------------+---------------------------------
#   pretrain | finetune_joint_k562_tss_sc_decima | job 11838935 (acc_mlm=1)
#   scratch  | THIS SCRIPT = job 11839840         | job 11839705 (acc_mlm=1)
#
# Differs from finetune_joint_k562_tss_sc_decima.sh only in that
# train.pretrained_model_path is commented out, so train.py:924 skips the load and the
# backbone is randomly initialised. Accessibility is a normal input here (acc_mlm=0).
#
# The point of the 2x2 is the interaction: pretraining is "predict masked accessibility from
# sequence", so it should help most where accessibility is withheld. This cell is the control
# that makes that comparison possible.
#
# CAVEAT: optimizer.lr=1e-4 was tuned for finetuning, not random init. A weak result here is
# ambiguous between "pretraining matters" and "1e-4 is wrong for scratch". Consider a short
# probe first -- trainer.limit_train_batches=0.2 makes an epoch ~2 h instead of ~9.6 h.
#
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
 +dataset.tss_json_file=/data1/lesliec/sarthak/data/DE_danwei/k562_sc_rna_info_cellranger.json \
 dataset.acc_type=continuous \
 dataset.acc_mlm=0 \
 +dataset.append_gene_mask=true \
 +dataset.upstream=163840 \
 \
 +model.config.skip_embedding=true trainer.devices=$NUM_GPUS \
 \
 task._name_=joint_tss \
 task.loss._name_=mse_tss \
 task.metrics=[mse_tss] \
 \
 decoder._name_=tss +decoder.pool_region=all +decoder.d_output=1 +decoder.hidden_dim=128 trainer.accumulate_grad_batches=16 \
#  train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2025-07-18/00-23-52-538795/checkpoints/last.ckpt"


# Single-GPU interactive version for a quick sanity check that the masking took effect --
# the JointMaskingEncoder print line at startup shows d_input2, and the first val numbers
# should be clearly worse than the ATAC run's if accessibility is really gone.
#
# CUDA_VISIBLE_DEVICES=0 pixi run python -m train wandb=null experiment=hg38/joint_finetune dataset.batch_size=1 \
#  trainer.precision=bf16 dataset.num_workers=1 loader.num_workers=1 model.config.vocab_size=1 model.config.pad_vocab_size_multiple=1 \
#  model=caduceus model.config.d_model=256 model.config.n_layer=16 model.config.bidirectional=true \
#  model._name_=dna_embedding_caduceus model.config.bidirectional_strategy=add model.config.bidirectional_weight_tie=true model.config.rcps=false \
#  optimizer.lr="1e-4" +train.remove_test_loader_in_eval=true \
#  dataset._name_=TSSLoader \
#  dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/K562_DNase.npz \
#  dataset.load_in=false dataset.shift_sequences=2000 +dataset.rc_strand=true \
#  +dataset.tss_json_file=/data1/lesliec/sarthak/data/DE_danwei/k562_sc_rna_info_cellranger.json \
#  dataset.acc_type=continuous dataset.acc_mlm=1 +dataset.mask_only=1 \
#  +dataset.append_gene_mask=true +dataset.upstream=163840 \
#  +model.config.skip_embedding=true trainer.devices=1 \
#  task._name_=joint_tss task.loss._name_=mse_tss task.metrics=[mse_tss] \
#  decoder._name_=tss +decoder.pool_region=all +decoder.d_output=1 +decoder.hidden_dim=128 \
#  train.pretrained_model_path="/data1/lesliec/sarthak/caduceus/outputs/2025-07-18/00-23-52-538795/checkpoints/last.ckpt"

# Resume: train.py does trainer.fit(model, ckpt_path=config.train.ckpt), so a run can be
# continued with train.ckpt=<path>/last.ckpt. There is no auto-detection here because hydra
# writes to a fresh timestamped outputs/ dir each launch -- which is also why this uses the
# non-preemptable lesliec,gpu partition rather than preemptable.
