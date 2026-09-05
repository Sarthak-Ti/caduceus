#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=150G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=e2g_3prime
#SBATCH --output=jobs/%x_%j.out

# E2G for the bp-resolution 3' RNA-seq profile model from job 6061182
# (slurm_scripts/finetune_joint_k562_3prime.sh).
#
# No --array: this is one checkpoint, one assay. And no --center_tss flag -- centering at the TSS is
# the only mode e2g_3prime.py has, because the gene-body / 3'-terminal aggregation windows are defined
# in coordinates relative to the TSS-centered window.
#
# Checkpoint: now the FIXED run (id 224, outputs/2026-08-13/16-16-44-251024), retrained on the
# rescaled K562_rnaseq_stranded_fixed.npz. Config is otherwise byte-identical to 217 below, and
# e2g_3prime.py nulls additional_tracks anyway, so the npz swap needs no change here.
#
# Epoch 7, not the val-loss minimum (epoch 2, 0.12826). The fixed run overfits almost immediately --
# train loss falls monotonically while val flattens, so the generalization gap grows from 0.10 at
# ep 2 to 0.20 at ep 7 to 0.28 by ep 12. But ep 7's val loss (0.13696) is only 0.009 off the
# minimum against a val sd of ~0.021 across epochs 1-14, i.e. indistinguishable, while having seen
# 3.5x more training -- which matters for a perturbation-response assay like E2G, where the
# quantity of interest is sensitivity to accessibility knockdown rather than absolute profile fit.
# Do NOT carry epoch 11 over from the 217 run below: for the fixed run that is a clearly overfit
# checkpoint (0.15093).
#
# val_loss is NOT comparable between the two runs: poisson_loss_nan is computed directly against
# the target, so the fixed run's ~0.13 vs 217's ~0.85 is the target rescale, not a 6x improvement.
#
# previous round -- 217, trained on the un-rescaled npz. That run declined slowly and steadily, so
# its best is epoch 14 (0.85225), not the epoch 11 that was scored while it was still training:
#   outputs/2026-08-05/13-39-56-645028/checkpoints/14-val_loss=0.85225.ckpt   (-o k562_3prime_e2g_ep14)
#   outputs/2026-08-05/13-39-56-645028/checkpoints/11-val_loss=0.85532.ckpt   (-o k562_3prime_e2g, already scored)
#
# Runtime: 2084 unmasked passes (one per gene, cached across that gene's elements) + 10280 masked
# passes over a 524288bp window. --load_data keeps the genome and K562_DNase npz in memory, which is
# what the 150G is for; the 3' RNA target npz is NOT loaded (e2g_3prime.py nulls additional_tracks,
# since only predictions are read).
#
# --batch_size 2 is conservative. The decoder's final_pointwise holds a (B, 524288, 512) fp32
# intermediate, ~1.1GB per element, so an 80GB A100 has room for considerably more -- raise it if the
# job is slower than expected.

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

CKPT="/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-16-44-251024/checkpoints/07-val_loss=0.13696.ckpt"

pixi run python -u e2g_3prime.py \
  -o "k562_3prime_fixed_e2g_ep7" \
  --ckpt_path "$CKPT" \
  --scale_factor 100 \
  --dist_additional_mask 100 \
  --three_prime_len 10000 \
  --tss_len 2000 \
  --batch_size 2 \
  --load_data
