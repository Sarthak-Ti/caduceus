#!/bin/bash
#SBATCH --partition=lesliec,gpu
#SBATCH --account=lesliec
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=100G
#SBATCH --time=168:00:00
#SBATCH --job-name=k562_cage103_to_3prime
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-k562_cage103_to_3prime.out
set -euo pipefail
cd /data1/lesliec/sarthak/caduceus
CAGE_FT_ROOT="$PWD/outputs/cage103_to_3prime_20260922"
CAGE_FT_CONFIG="$PWD/configs/transfer"
CAGE_FT_PYTHON="$PWD/.pixi/envs/default/bin/python"
# multiprocessing's Unix socket names must stay below the OS path-length limit.
CAGE_FT_TMP="/data1/lesliec/sarthak/tmp/c103_${SLURM_JOB_ID}"
mkdir -p "$CAGE_FT_TMP"
export TMPDIR="$CAGE_FT_TMP"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Prevent two submissions from training into the same checkpoint directory.
exec 9>"$CAGE_FT_ROOT/.training.lock"
flock -n 9 || { echo 'A CAGE103 transfer job already holds this run directory.'; exit 1; }
write_status() {
  "$CAGE_FT_PYTHON" - "$CAGE_FT_ROOT/status.json" "$1" "$SLURM_JOB_ID" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
p = Path(sys.argv[1])
value = dict(status=sys.argv[2], slurm_job_id=sys.argv[3],
             updated_utc=datetime.now(timezone.utc).isoformat(),
             training_dir=str(p.parent/'training'), max_epochs=40)
tmp = p.with_suffix('.json.tmp')
tmp.write_text(json.dumps(value, indent=2)+'\n')
tmp.replace(p)
PY
}
trap 'write_status failed' ERR
write_status validating_transfer
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$CAGE_FT_PYTHON" tests/validate_cage103_poisson_transfer.py \
  --config "$CAGE_FT_CONFIG/k562_cage103_poisson.yaml" \
  --output-dir "$CAGE_FT_ROOT/validation"

# Real full-length forward/backward, optimizer and validation checks on both GPUs.
# Smoke checkpoints are isolated; the full run reloads CAGE with a fresh optimizer.
write_status gpu_smoke_test
srun --cpu-bind=none --kill-on-bad-exit=1 "$CAGE_FT_PYTHON" -u -m train \
  --config-path "$CAGE_FT_CONFIG" --config-name k562_cage103_poisson \
  "hydra.run.dir=$CAGE_FT_ROOT/smoke_${SLURM_JOB_ID}" \
  wandb=null trainer.max_epochs=1 trainer.limit_train_batches=2 \
  trainer.limit_val_batches=2 trainer.accumulate_grad_batches=1 \
  train.test=false >"$CAGE_FT_ROOT/smoke_${SLURM_JOB_ID}.log" 2>&1
"$CAGE_FT_PYTHON" - "$CAGE_FT_ROOT/smoke_${SLURM_JOB_ID}" <<'PY'
import json, math, sys
from pathlib import Path
import torch
root = Path(sys.argv[1])
c = torch.load(root/'checkpoints/last.ckpt', map_location='cpu', weights_only=False)
assert c['global_step'] == 2, c['global_step']
assert all(torch.isfinite(v).all() for v in c['state_dict'].values())
scores = [float(v['current_score']) for v in c['callbacks'].values()
          if isinstance(v, dict) and v.get('current_score') is not None]
assert scores and all(math.isfinite(x) for x in scores), scores
source = torch.load(c['hyper_parameters']['train']['pretrained_model_path'],
                    map_location='cpu', weights_only=False)['state_dict']
changed = {prefix: sum(not torch.equal(v, source[k]) for k, v in c['state_dict'].items()
                       if k.startswith(prefix)) for prefix in ['encoder.', 'model.']}
assert all(n > 0 for n in changed.values()), changed
result = dict(status='passed', optimizer_steps=c['global_step'], validation_loss=scores,
              changed_state_tensors=changed)
(root/'smoke_check.json').write_text(json.dumps(result, indent=2)+'\n')
print('GPU smoke test passed:', json.dumps(result), flush=True)
PY

echo 'Starting full CAGE103-to-3prime fine-tuning (40 epochs, effective batch 32).'
write_status training
srun --cpu-bind=none --kill-on-bad-exit=1 "$CAGE_FT_PYTHON" -u -m train \
  --config-path "$CAGE_FT_CONFIG" --config-name k562_cage103_poisson \
  >"$CAGE_FT_ROOT/training.log" 2>&1
write_status complete
echo 'CAGE103-to-3prime training finished.'
