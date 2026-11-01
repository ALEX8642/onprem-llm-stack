#!/usr/bin/env bash
# Start Qwen3.8-Flash-Next-NVFP4 (community RadixArk quant, SGLang) on both Spark nodes (TP=2).
# TRIAL MODEL — evaluated 2026-08-28 (docs/MODELS.md). Recipe recovered from Spark Arena's
# public `/api/recipes/<id>/raw` endpoint (benchmark id 1bee0549-a211-4fe9-9c86-913ebf22d25a,
# 35.9 tok/s decode measured there) — not in any sparkrun registry, so it's run by file path
# rather than @handle, same as ds4flash0731. Public source: MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks.
#
# Recipe: scripts/recipes/qwen38-flash-next-nvfp4-sglang.yaml. Uses a public prebuilt image
# (ghcr.io/thupalo/qwen38-flashnext-sm121:20260827) — unlike ds4flash0731, no local Dockerfile
# patch needed.
#
# Quant: RadixArk/Qwen3.8-Flash-Next-NVFP4 — NVFP4 W4A4 on routed MoE experts only, everything
# else (attention/QSA/GDN/mHC/router/embeddings/MTP) stays BF16. Quantizer's own published evals:
# GSM8K 97.27 vs BF16 97.12-97.50 (no loss), AIME26 98.75 pass@1 vs BF16 100 (modest, recovers at
# majority@8). No coding-specific eval published — that's what this trial is for.
#
# sparkrun recipe vram: 79.5 GB/node model+KV at the recipe's declared 262144 context (2.2x
# headroom to 587,202 max tokens) — comfortable fit, unlike GLM-5.3-Flash's near-zero margin.
#
# Weights are NOT on modelstore yet (checked 2026-08-28) — first run pulls ~135GB from HF
# directly into the hot cache on each node, same as ds4flash0731's first run did.
#
# Memory budget: mem_fraction_static kept at the recipe's own tested default, 0.78 — NOT
# tuned down like MiniMax (0.70 floor) / ds4flash (0.85->0.80). Tried cutting it first (this
# model's real overhead looked similar in shape): 0.68 failed outright — SGLang refused to
# boot ("Loaded weights leave no GPU memory for the KV cache... Raise --mem-fraction-static
# above 0.703") — and 0.75 booted but the KV pool (173,824 tokens) came out SMALLER than the
# recipe's declared 262,144 context, i.e. a real capability cut, not a free one. Root cause:
# NEXTN speculative decoding's draft weights get counted against this budget too, and eat
# nearly all of the gap between the model's own floor (0.703) and its tested default (0.78) —
# unlike ds4flash/MiniMax, there's no real slack here to trim. 0.78 gives a genuine 337,024-
# token KV pool (1.29x the declared context). max_running_requests dropped 16->4 (one per
# family member, matching MiniMax's reasoning) — that one's a real, safe cut, unrelated to
# the mem-fraction floor. Full derivation: docs/MODELS.md, 2026-08-28.
#
# HF_HUB_OFFLINE=0 works around a real bug: SGLang's config loader runs a speculative
# `_try_load_longcat_config` probe against HF Hub even under offline mode (auto-set once
# weights are cached), and that probe isn't itself cached — hard-fails instead of skipping
# gracefully without this override.

set -euo pipefail

export PATH="$HOME/.local/share/uv/tools/sparkrun/bin:$PATH"

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$SCRIPTS_DIR/recipes/qwen38-flash-next-nvfp4-sglang.yaml"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

HEAD="head"
PORT=8010  # distinct from the existing :8000-8003 vLLM range and worker's :8004-8006/:8888
GPU_MEM=0.78
MAX_NUM_SEQS=4
# Context: NOT capped below the recipe's native 262144 (unlike MiniMax's 96k / ds4flash's
# 128k, both of which cut below their model's max). Checked the actual official technical
# report (Qwen3.8-Flash-Next tech_report.pdf, Table 3 — RULER + 8-needle MRCR, the QSA
# attention row): RULER 99.62 and MRCR 93.00 at the 128-256K bucket (covers our native
# 262144 ceiling) — both still strong. The real cliff is 256K->512K specifically: MRCR falls
# 93.00->40.53 there (RULER stays fine through 512K). That cliff sits in YaRN-extended
# territory this recipe doesn't enable, past our native ceiling — so running at full native
# context isn't the gamble an earlier pass through this assumed (that pass capped at 131072
# off an informal, unverified community claim — corrected 2026-08-28 after finding the
# primary-source table). Revisit only if real usage disagrees with the published numbers.
MAX_MODEL_LEN=262144
TIMEOUT=10800  # 3 hours — first run pulls ~135GB of weights, no local staging yet

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

LOGFILE="/tmp/qwen38-flash-next-nvfp4-$(date +%Y%m%d-%H%M%S).log"
log "Starting Qwen3.8-Flash-Next-NVFP4 (SGLang, TP=2, port $PORT)..."
log "Log: $LOGFILE"
echo ""

PYTHONUNBUFFERED=1 sparkrun run "$RECIPE" \
    --port "$PORT" \
    --gpu-mem "$GPU_MEM" \
    --max-model-len "$MAX_MODEL_LEN" \
    -o max_num_seqs="$MAX_NUM_SEQS" \
    -e HF_HUB_OFFLINE=0 \
    -e TRANSFORMERS_OFFLINE=0 \
    -e HF_HUB_ETAG_TIMEOUT=5 \
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
