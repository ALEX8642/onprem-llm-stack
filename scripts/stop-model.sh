#!/usr/bin/env bash
# Stop all running sparkrun models and verify both nodes are idle.
# Uses sparkrun status as the authoritative state check (nvidia-smi memory
# queries are not supported on GB10 unified memory architecture).

set -euo pipefail

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Stopping all sparkrun models..."
sparkrun stop --all 2>/dev/null || true
sleep 5

log "Waiting for both nodes to report idle..."
for i in $(seq 1 24); do
    STATUS=$(sparkrun status 2>/dev/null || echo "error")

    if echo "$STATUS" | grep -q "No sparkrun containers running"; then
        log "Both nodes idle. Safe to start a new model."
        exit 0
    fi

    log "  Still running... (attempt $i/24)"
    sleep 5
done

log "WARNING: Models did not stop within 2 minutes."
log "Run 'sparkrun status' to check what's still running."
log "You may need to manually run: sparkrun stop --force"
exit 1
