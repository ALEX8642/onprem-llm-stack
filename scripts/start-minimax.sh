#!/usr/bin/env bash
# Start MiniMax M2.7 NVFP4 on both Spark nodes (TP=2).
# Streams sparkrun output live, fails fast on known error patterns, and
# surfaces KV/VRAM stats on success. Run in screen/tmux if you want the
# model to survive terminal close.

set -euo pipefail

export PATH="$HOME/.local/share/uv/tools/sparkrun/bin:$PATH"

HEAD="head"
PORT=8002
TIMEOUT=1800  # 30 min — nightly image rebuild (10 min) + distribute to worker (10+ min) + torch.compile

# Verify the 200GbE interconnect is reachable before starting — fail fast rather
# than letting sparkrun time out after 30 min with a cryptic NCCL error.
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes ai@worker-cx7 true 2>/dev/null; then
    log "ERROR: Cluster interconnect unreachable (worker-cx7)."
    log "Check the physical link or run: ssh ai@worker 'ip link show enp1s0f0np0'"
    exit 1
fi

LOGFILE="/tmp/minimax-$(date +%Y%m%d-%H%M%S).log"
SPARKRUN_PID=""
TAIL_PID=""
MODEL_READY=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

on_exit() {
    local code=$?
    [ -n "$TAIL_PID" ] && kill "$TAIL_PID" 2>/dev/null || true
    if [ "$code" -ne 0 ] && [ -n "$SPARKRUN_PID" ]; then
        kill "$SPARKRUN_PID" 2>/dev/null || true
        # This is a TP=2 job spanning both nodes — killing the local sparkrun
        # client alone doesn't tear down worker's shard. Reuse the same
        # teardown stop-model.sh already proves safe, so an interrupted launch
        # (killed script, closed screen session, etc.) doesn't leave the
        # cluster in a stuck non-idle state that the next start/switch has to
        # be manually cleaned up before it can proceed.
        sparkrun stop --all 2>/dev/null || true
        [ "$MODEL_READY" -eq 0 ] && log "Full startup log: $LOGFILE"
    fi
}
trap on_exit EXIT
# EXIT alone is not reliably sufficient for signal-induced termination — the
# explicit converts-to-exit pattern below is what actually guarantees on_exit
# runs on a killed script, closed screen/tmux session (SIGHUP), or Ctrl-C
# (SIGINT), not just a normal `exit` call. Verified empirically 2026-08-03:
# a bare `trap on_exit EXIT` did NOT reliably fire on SIGTERM sent to a
# script blocked in `sleep` inside the polling loop below.
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

log "Starting MiniMax M2.7 NVFP4 (TP=2, port $PORT)..."
log "Log: $LOGFILE"
echo ""

# 0.70 required: vLLM nightly (v0.21.0+) CUDA graph memory profiling consumes ~4 GB per GPU,
# dropping effective utilization at 0.65 below what one 96k sequence needs (5.81 vs 5.66 GiB).
# At 0.70 the KV pool is 193,552 tokens — enough for 4 concurrent family users at typical
# context sizes (10–30k each) with room to spare before the context trimmer fires at 85k.
#
# max_model_len=98304: hard 96k cap. MiniMax quality degrades measurably past this point.
# Requests over the limit get a vLLM 400 — the OpenWebUI context_trimmer filter handles
# graceful compaction before hitting this wall.
#
# max_num_seqs=4: allow up to 4 concurrent requests (one per family member).
# vLLM uses continuous batching and incremental KV block allocation, so requests aren't
# pre-allocated max_model_len each — they share the pool dynamically. If the pool genuinely
# fills, vLLM preempts the lowest-priority sequence rather than erroring.
# Using max_num_seqs=2 would queue users 3 and 4 behind active generations, potentially
# hitting the LiteLLM 300s timeout before their response even starts.
PYTHONUNBUFFERED=1 sparkrun run @official/minimax-m2.7-nvfp4-vllm \
    --tp 2 --port "$PORT" --gpu-mem 0.70 \
    -o enable_prefix_caching=true \
    -o max_model_len=98304 \
    -o max_num_seqs=4 \
    -o enable_auto_tool_choice=true \
    -o tool_call_parser=minimax_m2 \
    -o reasoning_parser=minimax_m2 \
    > "$LOGFILE" 2>&1 200>&- &   # 200>&-: don't inherit switch-model.sh's lock (see below)
SPARKRUN_PID=$!

tail -f "$LOGFILE" 200>&- &
TAIL_PID=$!

START=$(date +%s)

while true; do
    ELAPSED=$(( $(date +%s) - START ))

    # Hard timeout — NCCL hang or image download stall
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo ""
        log "ERROR: Timed out after ${TIMEOUT}s."
        log "Check NCCL connectivity: ssh ai@worker-cx7 'ping -c1 head-cx7'"
        exit 1
    fi

    # Sparkrun process died — surface root error immediately
    if ! kill -0 "$SPARKRUN_PID" 2>/dev/null; then
        sleep 1  # Let tail flush remaining lines
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

    # KV cache too small for max_model_len — fail fast with a hint
    if grep -qE "EngineCore failed|ValueError: To serve|KV cache memory.*larger than|not enough kv cache" \
            "$LOGFILE" 2>/dev/null; then
        sleep 2
        echo ""
        log "ERROR: KV cache insufficient. Relevant lines:"
        grep -E "(ValueError|RuntimeError):|Available KV cache|KV cache.*needed|estimated maximum model length" \
            "$LOGFILE" 2>/dev/null | tail -6
        echo ""
        log "Fix: increase --gpu-mem or decrease max_model_len."
        exit 1
    fi

    # NCCL / inter-node comms failure
    if grep -qE "ncclInternal|NCCL.*error|Watchdog caught|Connection refused.*${PORT}" \
            "$LOGFILE" 2>/dev/null; then
        sleep 2
        echo ""
        log "ERROR: NCCL / inter-node communication failure."
        grep -E "NCCL|ncclInternal|Watchdog|Connection refused" "$LOGFILE" 2>/dev/null | tail -5
        exit 1
    fi

    # FlashInfer autotuner crash (seen on MTP recipes, not NVFP4, but guard anyway)
    if grep -qE "FlashInfer.*crash|autotuner.*failed|triton.*error" \
            "$LOGFILE" 2>/dev/null; then
        sleep 2
        echo ""
        log "ERROR: FlashInfer autotuner failure."
        log "Workaround: add '-o attention_backend=triton_attn' to the sparkrun command."
        grep -E "FlashInfer|autotuner|triton" "$LOGFILE" 2>/dev/null | tail -5
        exit 1
    fi

    # Success: API is responding
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
        # Surface the stats that matter most from startup (exact log line prefixes only)
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
