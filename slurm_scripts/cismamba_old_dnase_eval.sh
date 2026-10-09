#!/bin/bash
#SBATCH --partition=preemptable
#SBATCH --account=preemptable
#SBATCH --qos=preemptable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-%x.out
set -euo pipefail
cd /data1/lesliec/sarthak/caduceus
export PATH="/home/tiwaris2/.pixi/bin:$PATH"
export OMP_NUM_THREADS=4
kind=${1:?dsqtl or caqtl required}
checkpoint=$(pixi run --frozen python - <<'PY'
import json, os
from pathlib import Path
root = Path('/data1/lesliec/sarthak/caduceus')
run = root / 'outputs/cismamba_old_dnase_20260924/striped_gp128'
paths = list((run / 'checkpoints').glob('*-val_loss=*.ckpt'))
checkpoint = min(paths, key=lambda p: (float(p.stem.split('val_loss=')[1]), int(p.name.split('-')[0])))
print(checkpoint)
PY
)
case "$kind" in
  dsqtl)
    name=cismamba_old_dnase_20260924_striped_gp128.npy
    bash slurm_scripts/cismamba_speedup_dsqtl.sh "$checkpoint" "$name"
    pixi run --frozen python - "$checkpoint" "$name" <<'PY'
import json, sys
from pathlib import Path
p = Path('tests/results/cismamba_old_dnase_20260924/striped_gp128_old_dnase')
p.mkdir(parents=True, exist_ok=True)
c = Path(sys.argv[1])
record = dict(budget_epochs=30, checkpoint=str(c), selected_epoch_zero_based=int(c.name.split('-')[0]),
              filename_val_loss=float(c.stem.split('val_loss=')[1]),
              selection='minimum validation loss within 30 epochs',
              prediction='/data1/lesliec/sarthak/data/joint_playground/dsQTL/' + sys.argv[2])
(p / 'dsqtl_e30_selection.json').write_text(json.dumps(record, indent=2) + '\n')
PY
    exec pixi run --frozen python -m tests.score_cismamba_controls \
      --variant striped_gp128_old_dnase --budget 30 \
      --results-dir tests/results/cismamba_old_dnase_20260924
    ;;
  caqtl)
    exec bash slurm_scripts/cismamba_caqtl.sh striped_gp128_old_dnase "$checkpoint" \
      /data1/lesliec/sarthak/caduceus/tests/results/cismamba_old_dnase_20260924/caqtl
    ;;
  *) echo "Unknown benchmark: $kind" >&2; exit 2 ;;
esac
