#!/bin/bash
#SBATCH --partition=lesliec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=168:00:00
#SBATCH --mem=1G
#SBATCH --job-name=monitor_k562_striped
#SBATCH --output=/data1/lesliec/sarthak/caduceus/jobs/%j-monitor_k562_striped.out

source ~/.bashrc

JOB_NAME="finetune_k562plus10_striped_maskonly_3"
CADUCEUS_DIR="/data1/lesliec/sarthak/caduceus"
JOBS_DIR="$CADUCEUS_DIR/jobs"
SLURM_SCRIPT="$CADUCEUS_DIR/slurm_scripts/finetune_joint_k562_many_striped.sh"
STATE_FILE="$CADUCEUS_DIR/slurm_scripts/.monitor_k562_state"
LOG_FILE="$CADUCEUS_DIR/slurm_scripts/monitor_k562.log"
BASE_SEED=2222
INTERVAL=600  # 10 minutes

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# ── State helpers ─────────────────────────────────────────────────────────────

init_state() {
    if [ ! -f "$STATE_FILE" ]; then
        printf 'seed=%s\n' "$BASE_SEED" > "$STATE_FILE"
        log "Initialized state: seed=$BASE_SEED"
    fi
}

get_value() {
    grep "^$1=" "$STATE_FILE" 2>/dev/null | cut -d= -f2-
}

set_value() {
    local key="$1" val="$2"
    if grep -q "^${key}=" "$STATE_FILE" 2>/dev/null; then
        sed -i "s|^${key}=.*|${key}=${val}|" "$STATE_FILE"
    else
        echo "${key}=${val}" >> "$STATE_FILE"
    fi
}

# ── Checkpoint discovery ──────────────────────────────────────────────────────

# Parses all "dirpath has changed" lines from $1 and returns the most recently
# modified .ckpt file found across every checkpoint directory mentioned.
find_latest_ckpt() {
    local out_file="$1"

    local dirpaths
    dirpaths=$(grep "dirpath has changed" "$out_file" 2>/dev/null \
        | grep -oP "(?<=')[^']*checkpoints(?=')" \
        | sort -u)

    if [ -z "$dirpaths" ]; then
        log "No checkpoint directories found in $out_file"
        echo ""
        return 1
    fi

    local latest_ckpt="" latest_time=0

    while IFS= read -r dirpath; do
        [ -z "$dirpath" ] && continue
        if [ -d "$dirpath" ]; then
            while IFS= read -r ckpt_file; do
                local t
                t=$(stat -c %Y "$ckpt_file" 2>/dev/null || echo 0)
                if [ "$t" -gt "$latest_time" ]; then
                    latest_time=$t
                    latest_ckpt="$ckpt_file"
                fi
            done < <(find "$dirpath" -maxdepth 1 -name "*.ckpt" -type f 2>/dev/null)
        fi
    done <<< "$dirpaths"

    echo "$latest_ckpt"
}

# ── Job submission ────────────────────────────────────────────────────────────

# Submit a modified copy of SLURM_SCRIPT with updated train.seed and,
# optionally, train.ckpt.  Pass empty string for ckpt_raw to skip ckpt override.
submit_job() {
    local ckpt_raw="$1"   # raw filesystem path (literal =), empty = no override
    local seed="$2"

    if [ -n "$ckpt_raw" ]; then
        log "Submitting: seed=$seed  ckpt=$ckpt_raw"
    else
        log "Submitting: seed=$seed  (no ckpt override)"
    fi

    local tmp_script
    tmp_script=$(mktemp "$CADUCEUS_DIR/slurm_scripts/.tmp_submit_XXXXXX.sh")

    # Python handles substitution to avoid bash/sed escaping nightmares.
    python3 - "$SLURM_SCRIPT" "$ckpt_raw" "$seed" "$tmp_script" <<'PYEOF'
import sys, re

src, ckpt_raw, seed, dst = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

with open(src) as f:
    content = f.read()

# Replace train.seed=<digits>
content = re.sub(r'\btrain\.seed=\d+', f'train.seed={seed}', content)

if ckpt_raw:
    # Escape = as \= for hydra
    ckpt_hydra = ckpt_raw.replace('=', r'\=')
    content = re.sub(
        r'train\.ckpt="[^"]*"',
        lambda _: f'train.ckpt="{ckpt_hydra}"',
        content,
    )

with open(dst, 'w') as f:
    f.write(content)
PYEOF

    if [ $? -ne 0 ]; then
        log "ERROR: Python substitution failed"
        rm -f "$tmp_script"
        return 1
    fi

    chmod +x "$tmp_script"

    cd "$CADUCEUS_DIR" || { log "ERROR: cannot cd to $CADUCEUS_DIR"; rm -f "$tmp_script"; return 1; }

    local result exit_code
    result=$(sbatch "$tmp_script" 2>&1)
    exit_code=$?
    rm -f "$tmp_script"

    if [ "$exit_code" -eq 0 ]; then
        local job_id
        job_id=$(echo "$result" | awk '{print $4}')
        log "Submitted job $job_id. Incrementing seed to $((seed + 1))."
        set_value "seed" $((seed + 1))
        return 0
    else
        log "ERROR: sbatch failed: $result"
        return 1
    fi
}

# ── Single monitor iteration ──────────────────────────────────────────────────

check_once() {
    log "=== Monitor check ==="

    local current_seed
    current_seed=$(get_value "seed")

    # Is the target job queued or running?
    local running_info
    running_info=$(squeue -u "$(id -un)" --name="$JOB_NAME" \
        --format="%i %j %T" --noheader 2>/dev/null | head -1)

    if [ -z "$running_info" ]; then
        # ── Job is NOT running ────────────────────────────────────────────────
        log "Job '$JOB_NAME' is not running."

        local latest_out
        latest_out=$(ls -t "$JOBS_DIR"/*-"${JOB_NAME}".out 2>/dev/null | head -1)

        if [ -n "$latest_out" ]; then
            log "Latest output file: $latest_out"
            local latest_ckpt
            latest_ckpt=$(find_latest_ckpt "$latest_out")

            if [ -n "$latest_ckpt" ]; then
                log "Using checkpoint: $latest_ckpt"
                submit_job "$latest_ckpt" "$current_seed"
            else
                log "No checkpoint found; submitting with seed override only."
                submit_job "" "$current_seed"
            fi
        else
            log "No previous output file found; submitting fresh."
            submit_job "" "$current_seed"
        fi

    else
        # ── Job IS running ────────────────────────────────────────────────────
        local job_id job_state
        job_id=$(echo "$running_info" | awk '{print $1}')
        job_state=$(echo "$running_info" | awk '{print $3}')
        log "Job '$JOB_NAME' is running (ID=$job_id, state=$job_state)."

        local out_file="$JOBS_DIR/${job_id}-${JOB_NAME}.out"
        if [ ! -f "$out_file" ]; then
            log "Output file not yet available: $out_file"
            return 0
        fi

        local warning_count
        warning_count=$(tail -100 "$out_file" | grep -c "WARNING")
        log "WARNING lines in last 100: $warning_count"

        if [ "$warning_count" -gt 20 ]; then
            log "Excessive warnings ($warning_count > 20). Cancelling job $job_id..."
            scancel "$job_id"
            sleep 5

            local latest_ckpt
            latest_ckpt=$(find_latest_ckpt "$out_file")

            if [ -n "$latest_ckpt" ]; then
                log "Restarting with checkpoint: $latest_ckpt"
                submit_job "$latest_ckpt" "$current_seed"
            else
                log "No checkpoint found; restarting with seed override only."
                submit_job "" "$current_seed"
            fi
        fi
    fi

    log "=== Monitor check complete ==="
}

# ── Main loop ─────────────────────────────────────────────────────────────────

log "Monitor SLURM job started (PID=$$). Will check every $((INTERVAL/60)) minutes."
init_state

while true; do
    check_once
    sleep "$INTERVAL"
done
