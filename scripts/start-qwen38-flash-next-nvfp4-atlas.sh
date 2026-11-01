#!/usr/bin/env bash
# Start Qwen3.8-Flash-Next-NVFP4 on the Atlas engine (nvidia checkpoint, TP=EP=2) on both Spark nodes.
# PRODUCTION since 2026-10-24, replacing the SGLang build (start-qwen38-flash-next-nvfp4.sh, kept as
# the rollback: switch-model.sh qwen38flash). Same port (:8010) and served name, so LiteLLM, the
# watchdog and Auto Thinking are unchanged.
#
# Recipe: scripts/recipes/qwen38-flash-next-nvfp4-atlas.yaml (atlas registry recipe, image pinned).
# Why Atlas (A/B 2026-10-24, docs/MODELS.md): faster decode (67-73 vs 48-53 tok/s), ~20 GiB/node
# less RAM because the 51 GB PLE table is paged from NVMe instead of held in RAM, and FLUX fits
# alongside it. gpu-mem 0.60 still gives a 586K-token KV pool (SGLang at 0.78: 337K); the
# measured floor is about 0.58 (61.2 GB/node before KV).

set -euo pipefail

export PATH="$HOME/.local/share/uv/tools/sparkrun/bin:$PATH"

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$SCRIPTS_DIR/recipes/qwen38-flash-next-nvfp4-atlas.yaml"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

HEAD="head"
PORT=8010  # same as the SGLang build, so nothing downstream changes
GPU_MEM=0.60
MAX_NUM_SEQS=4
# Native context, as on SGLang (reasoning: start-qwen38-flash-next-nvfp4.sh).
MAX_MODEL_LEN=262144
TIMEOUT=3600  # ~2.5 min warm; a first run on a node without the weights syncs ~124 GiB

SPARKRUN_PID=""
TAIL_PID=""
MODEL_READY=0

on_exit() {
    local code=$?
    [ -n "$TAIL_PID" ] && kill "$TAIL_PID" 2>/dev/null || true
    if [ "$code" -ne 0 ] && [ -n "$SPARKRUN_PID" ]; then
        kill "$SPARKRUN_PID" 2>/dev/null || true
        # TP=2 spans both nodes — killing the local sparkrun client alone doesn't
        # tear down worker's shard. Same rationale as start-minimax.sh/start-ds4flash0731.sh.
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
    log "Check the physical link or run: ssh ai@worker 'ip link show enp1s0f0np0'"
    exit 1
fi

sparkrun recipe validate "$RECIPE"

LOGFILE="/tmp/qwen38-flash-next-nvfp4-atlas-$(date +%Y%m%d-%H%M%S).log"
log "Starting Qwen3.8-Flash-Next-NVFP4 (Atlas, TP=EP=2, port $PORT, gpu-mem $GPU_MEM)..."
log "Log: $LOGFILE"
echo ""

PYTHONUNBUFFERED=1 sparkrun run "$RECIPE" \
    --port "$PORT" \
    --gpu-mem "$GPU_MEM" \
    --max-model-len "$MAX_MODEL_LEN" \
    -o max_num_seqs="$MAX_NUM_SEQS" \
    -o max_batch_size="$MAX_NUM_SEQS" \
    > "$LOGFILE" 2>&1 200>&- &   # 200>&-: don't inherit switch-model.sh's lock (see below)
SPARKRUN_PID=$!

tail -f "$LOGFILE" 200>&- &
TAIL_PID=$!

START=$(date +%s)

while true; do
    ELAPSED=$(( $(date +%s) - START ))

    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo ""
        log "ERROR: Timed out after ${TIMEOUT}s."
        log "Check NCCL connectivity: ssh ai@worker-cx7 'ping -c1 head-cx7'"
        exit 1
    fi

    if ! kill -0 "$SPARKRUN_PID" 2>/dev/null; then
        sleep 1
        echo ""
        ROOT=$(grep -E "(ValueError|RuntimeError|Error):" "$LOGFILE" 2>/dev/null \
            | grep -v "UserWarning\|DeprecationWarning" | tail -5 || true)
        if [ -n "$ROOT" ]; then
            log "ERROR: sparkrun exited. Root cause:"
            echo "$ROOT"
        else
            log "ERROR: sparkrun exited with no diagnostic output — check full log."
        fi
        exit 1
    fi

    if grep -qE "EngineCore failed|ValueError: To serve|KV cache memory.*larger than|not enough kv cache|NV_ERR_NO_MEMORY|CUDA out of memory|torch.OutOfMemoryError" \
            "$LOGFILE" 2>/dev/null; then
        sleep 2
        echo ""
        log "ERROR: KV cache / memory insufficient. Relevant lines:"
        grep -E "(ValueError|RuntimeError):|Available KV cache|KV cache.*needed|estimated maximum model length|NV_ERR_NO_MEMORY|CUDA out of memory" \
            "$LOGFILE" 2>/dev/null | tail -6
        echo ""
        log "Fix: lower gpu_memory_utilization or max_model_len via -o overrides in this script."
        exit 1
    fi

    if grep -qE "ncclInternal|NCCL.*error|Watchdog caught|Connection refused.*${PORT}|RayActorError" \
            "$LOGFILE" 2>/dev/null; then
        sleep 2
        echo ""
        log "ERROR: NCCL / inter-node communication failure."
        grep -E "NCCL|ncclInternal|Watchdog|Connection refused|RayActorError" "$LOGFILE" 2>/dev/null | tail -5
        exit 1
    fi

    if curl -sf "http://${HEAD}:${PORT}/v1/models" >/dev/null 2>&1; then
        sleep 1
        kill "$TAIL_PID" 2>/dev/null || true
        TAIL_PID=""
        MODEL_READY=1
        # Release switch-model.sh's lock now that startup is done. It guards the stop/start, not
        # the model's lifetime; held forever, it made the dashboard/chat Switch and Stop report
        # "busy" whenever a model was up. No-op when this script is run directly (fd 200 unset).
        exec 200>&-
        echo ""
        log "Ready at http://${HEAD}:${PORT}/v1 (${ELAPSED}s)"
        echo ""
        grep -E "KV OVERCOMMIT|KV pool|free \(|Server live" "$LOGFILE" 2>/dev/null | tail -4 \
            | sed 's/\x1b\[[0-9;]*m//g' || true
        echo ""
        wait "$SPARKRUN_PID"
        exit $?
    fi

    sleep 3
done
