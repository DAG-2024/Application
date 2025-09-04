#!/usr/bin/env bash
set -euo pipefail

#############################
# Config (edit as needed)
#############################
HEAD_HOST="gpu-master"
REMOTE_USER="gefenra"
SBATCH_FILE="API.sbatch"

LOCAL_PORT=9000       # your laptop port
REMOTE_PORT=8000      # where Uvicorn listens on the compute node (127.0.0.1:8000)

POLL_INTERVAL=3       # seconds between squeue polls
ASSIGN_TIMEOUT=300    # seconds to wait for node assignment
STARTUP_GRACE=45       # seconds to give the app to start after node is assigned

HEALTH_TIMEOUT=120         # seconds to wait for /health
HEALTH_POLL_INTERVAL=2     # seconds between health polls

# If you always want gpu-01 and your sbatch pins it there, you can set:
# PIN_NODE="gpu-01"
PIN_NODE=""

#############################
# Helpers
#############################
err() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[*] $*"; }

# Run a command on the head node
remote() {
  ssh -o BatchMode=yes -o ControlMaster=auto -o ControlPersist=10m \
      -o ControlPath="$HOME/.ssh/cm-%C" \
      "${REMOTE_USER}@${HEAD_HOST}" "$@"
}

#############################
# 1) Submit the job
#############################
info "Submitting job: sbatch ${SBATCH_FILE} on ${HEAD_HOST}..."
SUBMIT_OUT="$(remote "sbatch ${SBATCH_FILE}" 2>&1 || true)"

# Expected: "Submitted batch job 3107"
JOB_ID="$(printf '%s\n' "$SUBMIT_OUT" | awk '/Submitted batch job/ {print $4}')"
[[ -n "${JOB_ID}" ]] || err "Failed to submit job or parse JobID. Output was:\n$SUBMIT_OUT"

info "Submitted JobID: ${JOB_ID}"

#############################
# 2) Find assigned node
#############################
TARGET_NODE=""

if [[ -n "$PIN_NODE" ]]; then
  TARGET_NODE="$PIN_NODE"
  info "Using pinned node: $TARGET_NODE"
else
  info "Waiting for node assignment (timeout: ${ASSIGN_TIMEOUT}s)..."
  SECS=0
  while [[ $SECS -lt $ASSIGN_TIMEOUT ]]; do
    # Query squeue for this job
    LINE="$(remote "squeue -j ${JOB_ID} -o '%.18i %.20j %.8T %.10M %R' -h" || true)"
    # Example LINE: "3107 DAG_API_Server RUNNING  0:32  gpu-01"
    TARGET_NODE="$(printf '%s\n' "$LINE" | awk '{print $5}')"
    STATE="$(printf '%s\n' "$LINE" | awk '{print $3}')"

    if [[ -n "$TARGET_NODE" && "$TARGET_NODE" != "N/A" && "$STATE" == "RUNNING" ]]; then
      info "Job is RUNNING on node: $TARGET_NODE"
      break
    fi

    sleep "$POLL_INTERVAL"
    SECS=$((SECS + POLL_INTERVAL))
  done

  [[ -n "$TARGET_NODE" && "$TARGET_NODE" != "N/A" ]] || err "Timed out waiting for node assignment."
fi

#############################
# 3) (Optional) quick remote check that the app is up
#############################
info "Giving the app a moment to start (${STARTUP_GRACE}s)..."
sleep "$STARTUP_GRACE"

# Just a best-effort check; ignore failures (maybe app takes longer)
remote "ssh -o BatchMode=yes -o StrictHostKeyChecking=no ${TARGET_NODE} 'ss -lntp | grep :${REMOTE_PORT} || true'" || true

#############################
# 4) Open the tunnel using ProxyJump
#############################
info "Opening tunnel: localhost:${LOCAL_PORT} -> ${TARGET_NODE}:127.0.0.1:${REMOTE_PORT} via ${HEAD_HOST}"
info "Press Ctrl+C to close the tunnel."

# If you have keys set up, this will be password-less. Otherwise you’ll be prompted twice (jump + node).
exec ssh -J "${REMOTE_USER}@${HEAD_HOST}" \
         -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
         "${REMOTE_USER}@${TARGET_NODE}"
