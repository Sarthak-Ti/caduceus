#!/bin/bash
#SBATCH --partition=lesliec
#SBATCH --account=lesliec
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=168:00:00
#SBATCH --requeue
#SBATCH --job-name=cismamba_speedup
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out
set -euo pipefail
cd /data1/lesliec/sarthak/caduceus
export PATH="/home/tiwaris2/.pixi/bin:$PATH"
export OMP_NUM_THREADS=4
export HYDRA_FULL_ERROR=1
variant=${1:?Specify candidate name}
stage=${2:-screen}
shift "$(( $# >= 2 ? 2 : 1 ))"
pool=128
width=256
blocks=16
mode=mamba_only
state=64
expand=2
headdim=64
case "$variant" in
  hydra_gp8) pool=8 ;;
  hydra_gp32) pool=32 ;;
  hydra_gp128) ;;
  striped_gp128) mode=striped; blocks=8 ;;
  striped_wide_gp128) mode=striped; blocks=8; width=512 ;;
  hydra_gp8_state128) pool=8; state=128 ;;
  hydra_gp8_expand1) pool=8; expand=1 ;;
  hydra_gp8_head32) pool=8; headdim=32 ;;
  hydra_gp8_deep24) pool=8; blocks=24 ;;
  hydra_gp4_expand1) pool=4; expand=1 ;;
  *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac
run_dir="${CISMAMBA_RUN_DIR:-$PWD/outputs/cismamba_speedup_20260918/${variant}_${stage}}"
extra=()
resume=checkpoints/last.ckpt
if [[ "$stage" == long ]]; then
  # Separate recovery snapshots from validation-ranked checkpoints used by dsQTL.
  resume="$run_dir/checkpoints/last.ckpt"
  if [[ -f "$run_dir/recovery/last.ckpt" && "$run_dir/recovery/last.ckpt" -nt "$resume" ]]; then
    resume="$run_dir/recovery/last.ckpt"
  fi
  extra+=(
    +callbacks.model_checkpoint_every_n_steps.monitor=null
    +callbacks.model_checkpoint_every_n_steps.save_top_k=1
    +callbacks.model_checkpoint_every_n_steps.save_last=true
    +callbacks.model_checkpoint_every_n_steps.every_n_train_steps=256
    +callbacks.model_checkpoint_every_n_steps.every_n_epochs=0
    +callbacks.model_checkpoint_every_n_steps.dirpath=recovery/
    '+callbacks.model_checkpoint_every_n_steps.filename="step-{step}"'
    +callbacks.model_checkpoint_every_n_steps.auto_insert_metric_name=false
  )
fi
if [[ "$stage" == smoke ]]; then
  extra+=(trainer.max_epochs=1 trainer.limit_train_batches=8 trainer.limit_val_batches=2)
else
  extra+=(trainer.max_epochs=30)
fi
exec pixi run --frozen python -u -m train \
  experiment=hg38/joint_pretrain model=striped_hydra wandb=null \
  "hydra.run.dir=$run_dir" train.test=false \
  +train.remove_test_loader_in_eval=true "train.ckpt=$resume" \
  trainer.devices=1 trainer.precision=bf16 dataset.batch_size=1 \
  dataset.batch_size_eval=1 trainer.accumulate_grad_batches=8 \
  dataset.num_workers=7 loader.num_workers=7 \
  model.config.d_model="$width" model.config.n_blocks="$blocks" \
  model.config.mode="$mode" model.config.expand="$expand" model.config.use_mem_eff_path=false \
  model.config.d_state="$state" model.config.headdim="$headdim" \
  +model.config.global_pooling="$pool" +model.config.d_in=64 \
  +encoder.transpose=true +encoder.norm=null +encoder.activation=null \
  optimizer.lr=1e-3 train.task2=reg train.custom_metric=poisson_loss_mask \
  dataset.acc_type=continuous dataset.acc_mlm=0.25 dataset.mlm=0 \
  +dataset.mask_only=true dataset.load_in=true \
  dataset.data_path=/data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz \
  +dataset.sequences_bed_file=/data1/lesliec/sarthak/data/DK_zarr/sequences_enformer.bed \
  "${extra[@]}" "$@"
