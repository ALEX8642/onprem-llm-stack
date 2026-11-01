#!/usr/bin/env bash
# Start DeepSeek-V4-Flash-0731 (community "dspark" port) on both Spark nodes (TP=2).
# TRIAL MODEL — evaluated 2026-08-03 (docs/MODELS.md), revisited 2026-08-07 after a
# spark-arena recipe surfaced (ghcr.io/anemll/dspark-vllm-gx10:0.1.1). Unlike MiniMax/
# Qwen3.5, this is an unofficial community container (39-star repo, 3 weeks old, 3 open
# correctness bugs, no published soak-test data) — see docs/MODELS.md for the full
# writeup before trusting this for anything unattended.
#
# Recipe: scripts/recipes/ds4flash-0731.yaml (pinned local copy of the spark-arena
# recipe — not in any sparkrun registry, so it's run by file path rather than @handle).
# cluster_only: true — there is no single-node fallback; this always needs both nodes.
#
# NOTE: this uses the OFFICIAL deepseek-ai/DeepSeek-V4-Flash-0731 weights, not the
# "Abliterated-Uncensored" fork the upstream repo's own README benchmarks were run
# against. Do not repoint DSPARK/model at that fork on this box.
#
# The upstream image has two bugs (broken ENTRYPOINT for non-serve containers, and a
# missing `ray` package) that make it fail before ever loading the model — see
# scripts/recipes/Dockerfile.ds4flash-fix and docs/MODELS.md for the full diagnosis.
# This script builds/reuses a locally-patched image (ds4flash-fixed:0.1.1) on both
# nodes rather than the raw upstream tag.

set -euo pipefail

export PATH="$HOME/.local/share/uv/tools/sparkrun/bin:$PATH"

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$SCRIPTS_DIR/recipes/ds4flash-0731.yaml"
FIX_DOCKERFILE="$SCRIPTS_DIR/recipes/Dockerfile.ds4flash-fix"
IMAGE="ds4flash-fixed:0.1.1"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

ensure_fixed_image() {
    local host="$1"
    if [ "$host" = "local" ]; then
        docker image inspect "$IMAGE" >/dev/null 2>&1 && return
        log "Building $IMAGE locally..."
        docker build -t "$IMAGE" -f "$FIX_DOCKERFILE" "$(dirname "$FIX_DOCKERFILE")"
    else
        ssh "$host" "docker image inspect '$IMAGE'" >/dev/null 2>&1 && return
        log "Building $IMAGE on $host..."
        # shellcheck disable=SC2029
        ssh "$host" "mkdir -p /tmp/ds4flash-fix"
        scp "$FIX_DOCKERFILE" "$host:/tmp/ds4flash-fix/Dockerfile"
        ssh "$host" "cd /tmp/ds4flash-fix && docker build -t '$IMAGE' -f Dockerfile ."
    fi
}

ensure_tuning_cache() {
    # Standalone Triton/FP8-GEMM kernel autotuning, run once (idempotent — sparkrun
    # skips shapes it already has cached). Runs on ONE host only (hardware is
    # identical across nodes); results auto-mount into every subsequent `sparkrun run`.
    # Discovered 2026-08-08 after repeated OOMs during model startup: this pre-tunes
    # the FlashInfer SM120 sparse-MLA autotune (used by the DSpark draft model) so it
    # loads a cached config instead of computing it live during startup, removing one
    # of two memory spikes that were causing earlyoom to kill a Ray worker mid-boot.
    # Takes ~3.5 hours the first time (mostly the FP8 dense-GEMM search); MoE tuning
    # will report 18/18 failed — expected, this model uses DeepGEMM, not the generic
    # Triton fused_moe kernel this tool targets, so there's nothing for it to tune there.
    local cache_dir="$HOME/.cache/sparkrun/tuning/vllm"
    compgen -G "$cache_dir/N=2048,K=4096,*.json" >/dev/null 2>&1 && return
    log "Running one-time kernel autotuning (sparkrun tune vllm) — this takes ~3.5 hours..."
    sparkrun tune vllm "$RECIPE" --tp 2 --image "$IMAGE"
}

ensure_fixed_image local
ensure_fixed_image worker
ensure_tuning_cache

HEAD="head"
PORT=8000
# Context cap: recipe defaults to max_model_len=524288 (512K). Capped here the same
# way MiniMax is capped at 96k: below the point where this model family's quality
# visibly falls off. Unlike MiniMax's 96k (measured on this cluster), this 128K
# figure is extrapolated from sibling V4-Pro's published MRCR retrieval curve
# (94% @128K, 82% @512K, 66% @1M) — DeepSeek-V4-Flash-0731 itself has no published
# per-length numbers. Revisit if real usage suggests otherwise. This is a quality
# decision, independent of the memory budget below.
MAX_MODEL_LEN=131072
# Memory budget: 0.80, landing on ~2.00x concurrency at 131072 context — settled
# 2026-08-08 after root-causing the OOM saga (summary in docs/HISTORY.md "Models"; full trail: git show 9a1cf59:docs/HISTORY.md).
# The real fix was two-part: (1) ensure_tuning_cache above removes the FlashInfer
# sparse-MLA autotune spike, (2) 0.80 (down from the recipe's own headroom-heavy
# defaults) leaves enough unified-memory margin during the DeepGEMM warmup spike
# (2166 kernel variants, ~35s, still uncached — no tuning knob covers it) for
# worker's other resident services (Chatterbox/SearXNG/faster-whisper/etc.) to
# coexist without earlyoom killing a Ray worker mid-boot. Verified: boots cleanly
# and serves real completions with those services left running, no coexistence
# workaround needed. 0.70 (MiniMax's floor) was tried first and is NOT enough here —
# it produces "insufficient KV cache" errors at full 131072 context because model
# loading alone eats 79.31 GiB/node, leaving too little for a 131K KV cache at that
# utilization. Don't reuse MiniMax's number for this model; the math differs.
GPU_MEM=0.80
# Real usage is 1-2 concurrent family members, not the recipe's default 64. 2.00x
# concurrency at 0.80 already covers this with margin — tried computing down to
# 1x and found it lands very close to (or below) the ~5.6 GiB hard minimum vLLM
# needs to serve even ONE full-131072-token request, i.e. 1x risks refusing to
# start at all rather than just capping throughput. Not worth it for zero real gain.
MAX_NUM_SEQS=2
# First run pulls a brand-new container image on both nodes AND ~155GB of model
# weights to worker (head is pre-seeded from modelstore by this script below).
# Qwen3.5 (61GB) budgets 90 min for distribute+compile; this is ~2.5x the weight
# size on an unproven image, so budget generously. Subsequent runs are fast once
# both nodes have it cached.
TIMEOUT=10800  # 3 hours, first run only

MODELSTORE=/mnt/modelstore/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731
HF_CACHE="$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731"

SPARKRUN_PID=""
TAIL_PID=""
MODEL_READY=0

on_exit() {
    local code=$?
    [ -n "$TAIL_PID" ] && kill "$TAIL_PID" 2>/dev/null || true
    if [ "$code" -ne 0 ] && [ -n "$SPARKRUN_PID" ]; then
        kill "$SPARKRUN_PID" 2>/dev/null || true
        # TP=2 spans both nodes — killing the local sparkrun client alone doesn't
        # tear down worker's shard. Same rationale as start-minimax.sh/start-qwen3.sh.
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

# Pre-seed head's hot HF cache from modelstore, same pattern as switch-model.sh,
# so sparkrun doesn't re-download what's already sitting on the external SSD.
if [ -d "$MODELSTORE" ] && [ "$(ls -A "$MODELSTORE" 2>/dev/null)" ]; then
    if [ ! -d "$HF_CACHE" ] || [ -z "$(ls -A "$HF_CACHE" 2>/dev/null)" ]; then
        log "Pulling DeepSeek-V4-Flash-0731 from modelstore into hot cache..."
        mkdir -p "$(dirname "$HF_CACHE")"
        rsync -a --info=progress2 "$MODELSTORE/" "$HF_CACHE/"
    fi
else
    log "WARNING: not found in modelstore yet — sparkrun will pull ~155GB from HF directly."
fi

sparkrun recipe validate "$RECIPE"

LOGFILE="/tmp/ds4flash-$(date +%Y%m%d-%H%M%S).log"
log "Starting DeepSeek-V4-Flash-0731 (TP=2, port $PORT)..."
log "Log: $LOGFILE"
echo ""

PYTHONUNBUFFERED=1 sparkrun run "$RECIPE" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-mem "$GPU_MEM" \
    -o max_num_seqs="$MAX_NUM_SEQS" \
    --image "$IMAGE" \
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

    if grep -qE "EngineCore failed|ValueError: To serve|KV cache memory.*larger than|not enough kv cache|NV_ERR_NO_MEMORY|CUDA out of memory" \
            "$LOGFILE" 2>/dev/null; then
        sleep 2
        echo ""
        log "ERROR: KV cache / memory insufficient. Relevant lines:"
        grep -E "(ValueError|RuntimeError):|Available KV cache|KV cache.*needed|estimated maximum model length|NV_ERR_NO_MEMORY|CUDA out of memory" \
            "$LOGFILE" 2>/dev/null | tail -6
        echo ""
        log "Fix: raise GPU_MEM above $GPU_MEM or lower MAX_MODEL_LEN below $MAX_MODEL_LEN in this script."
        exit 1
    fi

    if grep -qE "ncclInternal|NCCL.*error|Watchdog caught|Connection refused.*${PORT}|RayActorError|Ray usage stats.*error" \
            "$LOGFILE" 2>/dev/null; then
        sleep 2
        echo ""
        log "ERROR: NCCL / Ray inter-node communication failure."
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
        grep -E \
            "Available KV cache memory:|GPU KV cache size:|Maximum concurrency|Model loading took|torch.compile took in total|init engine" \
            "$LOGFILE" 2>/dev/null \
            | sed 's/.*\] /  /' \
            | sed 's/(.*pid=[0-9]*) //' \
            || true
        echo ""
        wait "$SPARKRUN_PID"
        exit $?
    fi

    sleep 3
done
