#!/bin/bash
#SBATCH --partition=lesliec
#SBATCH --account=lesliec
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --job-name=cismamba_dsqtl
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out
set -euo pipefail
cd /data1/lesliec/sarthak/caduceus
export PATH="/home/tiwaris2/.pixi/bin:$PATH"
export OMP_NUM_THREADS=4
run_dir=${1:?Training output directory or checkpoint file required}
out_name=${2:?Unique output filename required}
if [[ "$run_dir" == *.ckpt ]]; then
  ckpt="$run_dir"
else
  ckpt=$(pixi run --frozen python - "$run_dir" <<'PY'
import re
import sys
from pathlib import Path
paths = list((Path(sys.argv[1]) / 'checkpoints').glob('*-val_loss=*.ckpt'))
if not paths:
    raise SystemExit('No validation-scored checkpoints found')
def score(p):
    return float(re.search(r'val_loss=(.+)\.ckpt$', p.name).group(1))
print(min(paths, key=score).resolve())
PY
  )
fi
test -f "$ckpt"
output_path="/data1/lesliec/sarthak/data/joint_playground/dsQTL/$out_name"
if [[ -e "$output_path" ]]; then
  echo "Output already exists; refusing to overwrite: $output_path" >&2
  exit 1
fi
echo "Selected checkpoint: $ckpt"
echo "Output: $output_path"
cd evals
exec pixi run --frozen python -u dsqtl_onemodel.py \
  --ckpt_path "$ckpt" -o "$out_name" --mask_size 1000 \
  --data_path /data1/lesliec/sarthak/data/DK_zarr/zarr_arrays/cell_type_arrays/GM12878_DNase.npz \
  --load_data
