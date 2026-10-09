#!/bin/bash

#SBATCH --partition=lesliec,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=e2g_tss_dn
#SBATCH --output=jobs/%x_%A_%a.out
#SBATCH --array=13

cd /data1/lesliec/sarthak/caduceus/evals
nvidia-smi

#-- dinucleotide-shuffle E2G for the TSS models, the sequence counterpart of evaluate_e2g_tss.sh --
#new output names (_dinuc / _dinuc_acc) so the accessibility-only results from evaluate_e2g_tss.sh
#are never overwritten. Only the count head is scored, unperturbed (col 0) vs perturbed (cols 1..).
#
#NOTE: the count head is a plain regression head, so these live on the natural-log target scale and
#CAN BE NEGATIVE -- score as a log-space difference, not a ratio. --save_all_shuffles keeps every
#individual shuffle (one scalar per row, ~1MB total) so the mean is not baked in as a geometric mean
#in expression space; outputs[:, 1:].mean(axis=1) reproduces the averaged 2-column layout exactly.
#
#--time is 168h, not the 12h of evaluate_e2g_tss.sh: the accessibility-only runs took ~9h20 at 2
#sequences per element, and 11 shuffles means 12 sequences per element.
#
#-- current round: task 13 --
#13 = scratch, shuffle arm. Same checkpoint as task 12; task 12 gave it the accessibility
#knockdown only (bare output name), so the sequence perturbation was never run on it. This is
#the missing cell: it makes scratch directly comparable to tasks 10/11 (noacc, noacc_scratch)
#and to task 1 (k562_decima_2_dinuc) under the identical shuffle protocol. Output is _dinuc
#suffixed, so the bare acc-only file from task 12 is not touched.
#Runtime: tasks 10/11 (shuffle, 12 seq/element) took 2d07h; --time=72:00:00 covers it.
#
#-- previous round: the K562 2x2 over (pretraining x accessibility-input), tasks 10-12 --
#All three cells that were still training as of 2026-09-14 live here rather than being split
#across this file and evaluate_e2g_tss.sh, because --perturbation acc in this script is
#numerically identical to e2g_tss.py for a single-output TSSDecoder: n_alt=1 gives the same 2-row
#batch, a[1:,0,s:e] /= scale_factor is the same slice as a[1,0,s:e] with 2 rows, the window/flip
#math is the same block, and average_shuffles is False so the output keeps the (n_pairs, 2)
#[before, after] layout. The only differences are bookkeeping: an extra _args.json sidecar, and
#--seed / --n_shuffles / --dist_additional_shuffle / --min_shuffle_len sitting inert.
#
#Which perturbation each cell gets is forced by what it was trained on:
#
#            | ATAC input                     | ATAC masked
#   ---------+--------------------------------+---------------------------
#   pretrain | k562_decima_2 (tasks 1/6)       | 10 = job 11838935 (shuffle)
#   scratch  | 12 = job 11839840 (acc)         | 11 = job 11839705 (shuffle)
#
#10 = noacc         : pretrained backbone (2025-07-18 last.ckpt), acc_mlm=1 + mask_only=1
#11 = noacc_scratch : random init, acc_mlm=1 + mask_only=1
#12 = scratch       : random init, accessibility as a normal input (acc_mlm=0)
#
#Tasks 10 and 11 were trained with acc_mlm=1 + mask_only=1, i.e. the accessibility track is masked
#at rate 1.0 with no BERT-style 10% random / 10% unchanged, so every position reaches the model as
#a zero with the mask channel set to 1. An accessibility knockdown is therefore a NO-OP on them --
#a[1:,0,s:e] /= scale_factor divides entries that are already exactly 0, leaving the perturbed
#column bit-identical to the unperturbed one. Dinucleotide shuffling the element is the only
#perturbation they can respond to, hence perturbation=shuffle and not both_perturbations.
#Task 12 does get real ATAC, so it takes the accessibility knockdown instead.
#
#All three are single-output TSSDecoder (decoder._name_=tss, pool_region=all, d_output=1) on the
#Decima geometry (gene mask input channel, upstream=163840, rc_strand on,
#k562_sc_rna_info_cellranger.json). e2g_tss_profile_dn.py builds the decoder from the config's
#_name_, so it handles them as-is -- same as it already does for decima_2 and tssmask below.
#
#COMPARISON NOTE: task 12 is scored under a different perturbation from tasks 10-11, so those
#numbers are not directly comparable to each other. The within-perturbation contrasts are the ones
#to read: 10 vs 11 isolates pretraining with accessibility withheld, and 10 vs task 1
#(k562_decima_2_dinuc, the pretrain/ATAC-input cell under the identical shuffle protocol) isolates
#withholding accessibility at fixed pretraining. Task 12 pairs with task 6
#(k562_decima_2_dinuc_acc) only loosely, since that one shuffles as well as knocking down.
#Task 12's output name is deliberately bare rather than _dinuc/_dinuc_acc suffixed: it is an
#accessibility-only run, the same 2-column quantity every bare-named file in the e2g dir holds.
#
#EPOCHS: each run at its own LOWEST val/loss epoch -- 04 / 19 / 27 respectively. Unlike the
#profile rounds elsewhere in these files, val/loss here IS the quantity the eval scores: all three
#use task.loss._name_=mse_tss with task.metrics=[mse_tss], and the logs confirm val/mse_tss equals
#val/loss exactly, so "lowest val/loss" needs no proxy argument and is comparable ACROSS the three
#(the loss is identical in form). Off-by-one reminder: the log's "Epoch e" progress-bar line reports
#what ModelCheckpoint saved as file e-1, so these filenames sit one below the log epochs.
#
#CONFOUND, read the numbers with it in mind: these are NOT matched training lengths.
#  noacc (pretrained)  min 1.53684 @ ep 04, then every one of the next 22 epochs is worse
#                      (1.833 @ ep 26) with train MSE at 0.0896 -- it memorised the training set
#                      almost immediately. Its "best" is a 4-epoch model.
#  noacc_scratch       min 1.79403 @ ep 19 (ep 16 is 1.79463, effectively tied), noisy throughout.
#  scratch (real ATAC) min 0.68757 @ ep 27, still drifting down slowly when training stopped.
#So a 4-epoch model is being compared against a 19- and a 27-epoch one. The pretrained run
#converging by epoch 5 and then overfitting hard, while the scratch runs were still descending, is
#itself the pretraining result -- but do not read the raw gap as a fixed-budget comparison. Rerun
#at a matched epoch (26 is the last checkpoint all three share) if that comparison is needed.
#
#TRAINING ENDED EARLY, all three cancelled 2026-09-14 as good enough rather than run to completion:
#jobs 11838935 and 11839705 had in fact been dead since 2026-09-13 (OSError Errno 28, /tmp full on
#iscb027 and iscb029 during dataloader worker startup), last checkpoint 26; job 11839840 was alive
#in epoch 32 and was cancelled deliberately. Set TMPDIR onto /data1 in the training script if these
#are ever rerun.
#-- previous round: five checkpoints x two perturbation arms --
#0/5 = decima_fixed_ep34 : TSSProfileDecoder, gene mask channel, Decima window, multinomial profile NLL (1e-6)
#1/6 = decima_2          : TSSDecoder (count only), gene mask channel, Decima window -- the matched
#                          count-only control for decima_fixed_ep34: identical dataset, window, mask
#                          channel and MSE count target, the profile head is the only difference
#2/7 = poisson_ep35      : TSSProfileDecoder, gene mask channel, Decima window, Poisson profile NLL (1.0)
#3/8 = tss_fixed_ep10    : TSSProfileDecoder, TSS mask channel, Decima window, multinomial profile NLL (1e-6)
#4/9 = tssmask           : TSSDecoder (count only), TSS mask channel, TSS-CENTERED window (TSS at
#                          262144, not 163840). Second profile-vs-count-only contrast, for the TSS
#                          mask channel -- but the window differs from tss_fixed_ep10, so that pair
#                          is confounded by geometry; it also gives the TSS-centered vs Decima
#                          window contrast against decima_2. Its in-context set is 3957 elements
#                          rather than the 3914 of the other four, so restrict to the intersection
#                          (in_context AND) before pooling metrics across all five.
#tasks 0-4 shuffle the element only, tasks 5-9 shuffle AND knock down accessibility.
outputs=( \
  "k562_tss_sc_rna_profile_decima_fixed_ep34_dinuc" \
  "k562_decima_2_dinuc" \
  "k562_tss_sc_rna_poisson_ep35_dinuc" \
  "k562_tss_sc_rna_profile_tss_fixed_ep10_dinuc" \
  "k562_tssmask_dinuc" \
  "k562_tss_sc_rna_profile_decima_fixed_ep34_dinuc_acc" \
  "k562_decima_2_dinuc_acc" \
  "k562_tss_sc_rna_poisson_ep35_dinuc_acc" \
  "k562_tss_sc_rna_profile_tss_fixed_ep10_dinuc_acc" \
  "k562_tssmask_dinuc_acc" \
  "k562_tss_sc_decima_noacc_ep04_dinuc" \
  "k562_tss_sc_decima_noacc_scratch_ep19_dinuc" \
  "k562_tss_sc_decima_scratch_ep27" \
  "k562_tss_sc_decima_scratch_ep27_dinuc" \
)

ckpts=( \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-11-23-053119/checkpoints/34-val_loss=0.95305.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-07-27/21-54-24-337889/checkpoints/16-val_loss=0.65628.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-10-59-441587/checkpoints/35-val_loss=0.72645.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-15-08-293121/checkpoints/10-val_loss=1.05699.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-07-27/23-00-33-044232/checkpoints/16-val_loss=0.68752.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-11-23-053119/checkpoints/34-val_loss=0.95305.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-07-27/21-54-24-337889/checkpoints/16-val_loss=0.65628.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-10-59-441587/checkpoints/35-val_loss=0.72645.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-08-13/16-15-08-293121/checkpoints/10-val_loss=1.05699.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-07-27/23-00-33-044232/checkpoints/16-val_loss=0.68752.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-09-08/11-32-12-827324/checkpoints/04-val_loss=1.53684.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-09-08/11-37-47-202762/checkpoints/19-val_loss=1.79403.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-09-08/13-23-33-047440/checkpoints/27-val_loss=0.68757.ckpt" \
  "/data1/lesliec/sarthak/caduceus/outputs/2026-09-08/13-23-33-047440/checkpoints/27-val_loss=0.68757.ckpt" \
)

#tasks 10-11 are shuffle-only: their accessibility is already fully masked, so the acc arm of
#both_perturbations would be a no-op. Task 12 is acc-only: it is the cell that gets real ATAC, and
#with --perturbation acc this script reduces exactly to e2g_tss.py
perturbations=("shuffle" "shuffle" "shuffle" "shuffle" "shuffle" \
               "both_perturbations" "both_perturbations" "both_perturbations" \
               "both_perturbations" "both_perturbations" \
               "shuffle" "shuffle" "acc" "shuffle")

#matches evaluate_e2g_striped_dn.sh so the shuffle noise floor is comparable across model families
N_SHUFFLES=11

#-- pick the right one based on SLURM_ARRAY_TASK_ID --
i=$SLURM_ARRAY_TASK_ID
OUTPUT=${outputs[$i]}
CKPT=${ckpts[$i]}
PERTURBATION=${perturbations[$i]}

echo "Running task $i: output=$OUTPUT, ckpt=$CKPT, perturbation=$PERTURBATION, n_shuffles=$N_SHUFFLES"
#-- run the task --
#--scale_factor and --dist_additional_mask are inert for perturbation=shuffle, only used by
#both_perturbations. --dist_additional_shuffle matches --dist_additional_mask so the sequence and
#accessibility perturbations cover the identical interval in the joint arm.
#--load_data still only loads the accessibility npz: e2g_tss_profile_dn.py overrides the profile
#model's expression_data_path to None, so the 12G RNA-seq npz is never opened
pixi run python -u e2g_tss_profile_dn.py \
  -o "$OUTPUT" \
  --ckpt_path "$CKPT" \
  --perturbation "$PERTURBATION" \
  --n_shuffles "$N_SHUFFLES" \
  --save_all_shuffles \
  --dist_additional_shuffle 100 \
  --scale_factor 100 \
  --dist_additional_mask 100 \
  --seed 0 \
  --batch_size 2 \
  --load_data \

#the five checkpoints, for reference (see the table in the notes for why these five):
#  decima_fixed_ep34  outputs/2026-08-13/16-11-23-053119/checkpoints/34-val_loss=0.95305.ckpt
#  decima_2           outputs/2026-07-27/21-54-24-337889/checkpoints/16-val_loss=0.65628.ckpt
#  poisson_ep35       outputs/2026-08-13/16-10-59-441587/checkpoints/35-val_loss=0.72645.ckpt
#  tss_fixed_ep10     outputs/2026-08-13/16-15-08-293121/checkpoints/10-val_loss=1.05699.ckpt
#  tssmask            outputs/2026-07-27/23-00-33-044232/checkpoints/16-val_loss=0.68752.ckpt
#  noacc              outputs/2026-09-08/11-32-12-827324/checkpoints/04-val_loss=1.53684.ckpt
#  noacc_scratch      outputs/2026-09-08/11-37-47-202762/checkpoints/19-val_loss=1.79403.ckpt
#  scratch            outputs/2026-09-08/13-23-33-047440/checkpoints/27-val_loss=0.68757.ckpt
#
#not run here:
#  k562_finetune_2 (2026-07-27) -- TSS-centered window AND no appended mask channel AND
#    pool_region='tss', so it varies three things at once against everything else here
#  k562_tss_sc_rna_profile2_ep31 (2026-08-05) -- superseded round, trained on the un-rescaled
#    K562_rnaseq_stranded.npz
#  k562_decima (2026-07-21) -- loads strict-clean but predates the TSSDecoder head change
#    (trailing Softplus + log2 removed); confirm with eval_tss.py before trusting it
#  k562_tss_simpledecoder / _sc (2026-03-11) -- simple=True went from Sequential(Linear, Softplus)
#    to a bare Linear, so these no longer load strict at all
