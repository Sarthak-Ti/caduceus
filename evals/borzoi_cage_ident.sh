#!/bin/bash

# cpu partition, not gpu: this is pure zarr IO plus numpy/scipy, no GPU is touched.
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --job-name=borzoi_cage_ident
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out

# Model-vs-every-cell-type CAGE correlations over borzoi fold3, strand-separate.
#   -> /data1/lesliec/sarthak/data/borzoi/borzoi_fold3_CAGE_ident.npz
#
#   sbatch evals/borzoi_cage_ident.sh
#
# MUST run in the caduceus env. borzoi_fold3_CAGE.zarr is zarr v2 and GM12878CAGE.zarr
# (read by the pre-gate) is v3; the borzoi pixi env ships zarr 2.x and cannot open the
# v3 store. This env has zarr 3.0.4, which reads both.
#
# Cost: 108 GB read at ~580 MB/s is ~3 min and is not the bottleneck; the per-region
# work is ~360 ms (two 638-track gaussian smooths dominate), so 6888 regions over 16
# workers is ~10 min. The 4 h limit is slack, not an estimate.
#
# --mem 64G: the parent holds ~750 MB of results, and each of 16 workers holds a
# BLOCK=8 slab (125 MB float16 + its float32 copy) plus the smoothing temporaries.

source ~/.bashrc
cd /data1/lesliec/sarthak/caduceus
mkdir -p jobs

pixi run python -u evals/borzoi_cage_ident.py -j "${SLURM_CPUS_PER_TASK:-16}" "$@"
