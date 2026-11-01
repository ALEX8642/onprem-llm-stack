#!/usr/bin/env bash
# Start Command A+ W4A4 on both Spark nodes (TP=2), port 8020. EXPERIMENT — branch exp/command-a-plus.
# Write-up and numbers: docs/experiments/command-a-plus.md. Run through `switch-model.sh command-a`
# (frees ComfyUI, stops the current model, passes the switch lock) rather than directly.
# Nothing in LiteLLM points at :8020: it's benchmarked directly, and the Qwen aliases (family chat,
# Home Assistant) are down while it runs. Restore with `switch-model.sh qwen38atlas` from ~/ai-stack.
#
# Recipe: scripts/recipes/command-a-plus-w4a4-vllm.yaml (by file path, not in any registry).
# Image:  command-a-plus-vllm:0.1 = pinned eugr/spark-vllm + cohere_melody (Dockerfile.command-a-plus),
#         built on each node the first time.

set -euo pipefail

export PATH="$HOME/.local/share/uv/tools/sparkrun/bin:$PATH"

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$SCRIPTS_DIR/recipes/command-a-plus-w4a4-vllm.yaml"
DOCKERFILE="$SCRIPTS_DIR/recipes/Dockerfile.command-a-plus"
IMAGE="command-a-plus-vllm:0.1"

HEAD="head"
PORT=8020
# Overridable for the memory-floor search; defaults are the settled values.
GPU_MEM="${GPU_MEM:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"   # the card's 128K input limit
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"          # one per family member, same as every other recipe here
TIMEOUT=3600  # first run copies ~132 GB to worker

log() { echo "[$(date '+%H:%M:%S')] $*"; }

ensure_image() {
    local host="$1"
    if [ "$host" = "local" ]; then
        docker image inspect "$IMAGE" >/dev/null 2>&1 && return
        log "Building $IMAGE locally..."
        docker build -t "$IMAGE" -f "$DOCKERFILE" "$(dirname "$DOCKERFILE")"
    else
        ssh "$host" "docker image inspect '$IMAGE'" >/dev/null 2>&1 && return
        log "Building $IMAGE on $host..."
        ssh "$host" "mkdir -p /tmp/command-a-plus-image"
        scp "$DOCKERFILE" "$host:/tmp/command-a-plus-image/Dockerfile"
        ssh "$host" "cd /tmp/command-a-plus-image && docker build -t '$IMAGE' -f Dockerfile ."
    fi
}

SPARKRUN_PID=""
TAIL_PID=""
MODEL_READY=0
LOGFILE="/tmp/command-a-plus-$(date +%Y%m%d-%H%M%S).log"

on_exit() {
    local code=$?
    [ -n "$TAIL_PID" ] && kill "$TAIL_PID" 2>/dev/null || true
    if [ "$code" -ne 0 ] && [ -n "$SPARKRUN_PID" ]; then
        kill "$SPARKRUN_PID" 2>/dev/null || true
        # TP=2 spans both nodes: killing the local client alone leaves worker's shard running.
        sparkrun stop --all 2>/dev/null || true
        [ "$MODEL_READY" -eq 0 ] && log "Full startup log: $LOGFILE"
    fi
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

if ! ssh -o ConnectTimeout=5 -o BatchMode=yes ai@worker-cx7 true 2>/dev/null; then
    log "ERROR: Cluster interconnect unreachable (worker-cx7)."
    exit 1
fi

ensure_image local
ensure_image worker

sparkrun recipe validate "$RECIPE"

log "Starting Command A+ W4A4 (TP=2, port $PORT, gpu-mem $GPU_MEM, ctx $MAX_MODEL_LEN x $MAX_NUM_SEQS)..."
log "Log: $LOGFILE"

PYTHONUNBUFFERED=1 sparkrun run "$RECIPE" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-mem "$GPU_MEM" \
    -o max_num_seqs="$MAX_NUM_SEQS" \
    --image "$IMAGE" \
    > "$LOGFILE" 2>&1 200>&- &   # 200>&-: don't inherit switch-model.sh's lock
SPARKRUN_PID=$!

tail -f "$LOGFILE" 200>&- &
TAIL_PID=$!

START=$(date +%s)
while true; do
    ELAPSED=$(( $(date +%s) - START ))

    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        log "ERROR: Timed out after ${TIMEOUT}s."
        exit 1
    fi

    if ! kill -0 "$SPARKRUN_PID" 2>/dev/null; then
        sleep 1
        log "ERROR: sparkrun exited. Last errors:"
        grep -E "(ValueError|RuntimeError|Error):" "$LOGFILE" 2>/dev/null \
            | grep -v "UserWarning\|DeprecationWarning" | tail -5 || true
        exit 1
    fi

    if grep -qE "EngineCore failed|ValueError: To serve|KV cache memory.*larger than|not enough kv cache|NV_ERR_NO_MEMORY|CUDA out of memory" \
            "$LOGFILE" 2>/dev/null; then
        sleep 2
        log "ERROR: KV cache / memory insufficient:"
        grep -E "(ValueError|RuntimeError):|Available KV cache|KV cache.*needed|estimated maximum model length|NV_ERR_NO_MEMORY|CUDA out of memory" \
            "$LOGFILE" 2>/dev/null | tail -6
        exit 1
    fi

    if grep -qE "ncclInternal|NCCL.*error|Watchdog caught|RayActorError" "$LOGFILE" 2>/dev/null; then
        sleep 2
        log "ERROR: NCCL / Ray inter-node failure:"
        grep -E "NCCL|ncclInternal|Watchdog|RayActorError" "$LOGFILE" 2>/dev/null | tail -5
        exit 1
    fi

    if curl -sf "http://${HEAD}:${PORT}/v1/models" >/dev/null 2>&1; then
        kill "$TAIL_PID" 2>/dev/null || true
        TAIL_PID=""
        MODEL_READY=1
        exec 200>&-   # release switch-model.sh's lock now that startup is done
        log "Ready at http://${HEAD}:${PORT}/v1 (${ELAPSED}s)"
        grep -E "Available KV cache memory:|GPU KV cache size:|Maximum concurrency|Model loading took|torch.compile took in total|init engine" \
            "$LOGFILE" 2>/dev/null | sed 's/.*\] /  /' || true
        wait "$SPARKRUN_PID"
        exit $?
    fi

    sleep 3
done
