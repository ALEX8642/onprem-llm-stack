#!/usr/bin/env bash
# Start GPT-OSS-120B (MXFP4) as the fallback model.
# This is your proven working model — use it when other models fail or during testing.
# Recipe: local/gpt  (uses ~/gpt.yaml)

set -euo pipefail

export PATH="$HOME/.local/share/uv/tools/sparkrun/bin:$HOME/.local/bin:$PATH"

HEAD="head"
PORT=8001

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Starting GPT-OSS-120B MXFP4 on both nodes (TP=2, port $PORT)..."
log "Expected load time: 3-5 minutes. This is your proven fallback."

sparkrun run gpt --tp 2 --port "$PORT" &
SPARKRUN_PID=$!

log "Waiting for model endpoint on :$PORT..."
for i in $(seq 1 60); do
    if curl -sf "http://${HEAD}:${PORT}/v1/models" >/dev/null 2>&1; then
        log "GPT-OSS-120B ready at http://${HEAD}:${PORT}/v1"
        log "LiteLLM 'fallback' alias will route here automatically."
        wait "$SPARKRUN_PID"
        exit 0
    fi
    sleep 5
done

log "ERROR: Model did not become healthy after 5 minutes."
kill "$SPARKRUN_PID" 2>/dev/null || true
exit 1
