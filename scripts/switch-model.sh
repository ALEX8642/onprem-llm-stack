#!/usr/bin/env bash
# Switch between models. Pulls weights from modelstore if not in local cache,
# then stops the current model and starts the new one.
# Usage: switch-model.sh minimax|qwen3.5|gpt-oss|ds4flash|qwen38flash|qwen38atlas|command-a

set -euo pipefail

# Guard against concurrent switches (e.g. one triggered via SSH while another is
# already running via the model-control API) — both would stop/start models at
# once with no coordination otherwise. Held until the new model reports ready (the start
# script inherits it via exec and closes it then).
LOCKFILE="/tmp/ai-stack-model-switch.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
    echo "A model switch is already in progress. Try again in a few minutes." >&2
    exit 1
fi

export PATH="$HOME/.local/share/uv/tools/sparkrun/bin:$PATH"

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=/dev/null
[ -f "$SCRIPTS_DIR/../.env" ] && source "$SCRIPTS_DIR/../.env"
MODELSTORE=/mnt/modelstore/huggingface/hub
HF_CACHE=$HOME/.cache/huggingface/hub

declare -A MODEL_CACHE_DIR=(
    [minimax]="models--nvidia--MiniMax-M2.7-NVFP4"
    [qwen3.5]="models--Qwen--Qwen3.5-122B-A10B-FP8"
    [gpt-oss]="models--openai--gpt-oss-120b"
    [ds4flash]="models--deepseek-ai--DeepSeek-V4-Flash-0731"
    [qwen38flash]="models--RadixArk--Qwen3.8-Flash-Next-NVFP4"
    [qwen38atlas]="models--nvidia--Qwen3.8-Flash-Next-NVFP4"
    [command-a]="models--CohereLabs--command-a-plus-05-2026-w4a4"
)

declare -A MODEL_SCRIPT=(
    [minimax]="start-minimax.sh"
    [qwen3.5]="start-qwen3.sh"
    [gpt-oss]="start-fallback.sh"
    [ds4flash]="start-ds4flash0731.sh"
    [qwen38flash]="start-qwen38-flash-next-nvfp4.sh"
    [qwen38atlas]="start-qwen38-flash-next-nvfp4-atlas.sh"
    [command-a]="start-command-a-plus.sh"
)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

usage() {
    echo "Usage: $0 <model>"
    echo "  minimax  — MiniMax M2.7 NVFP4         :8002  ~4-5 min  (primary, TP=2)"
    echo "  qwen3.5  — Qwen3.5-122B FP8           :8003  ~2-3 min  (TP=2)"
    echo "  gpt-oss  — GPT-OSS-120B MXFP4         :8001  (currently broken — image rebuild needed)"
    echo "  ds4flash — DeepSeek-V4-Flash-0731     :8000  TRIAL — unofficial community container,"
    echo "                                                see docs/MODELS.md before relying on it"
    echo "  qwen38atlas — Qwen3.8-Flash-Next-NVFP4 :8010  ACTIVE — nvidia checkpoint, Atlas engine"
    echo "  qwen38flash — Qwen3.8-Flash-Next-NVFP4 :8010  rollback — RadixArk quant, SGLang"
    echo "  command-a — Command A+ W4A4 (vLLM)      :8020  EXPERIMENT (docs/experiments/command-a-plus.md), no LiteLLM alias"
    exit 1
}

pull_from_modelstore() {
    local model="$1"
    local cache_dir="${MODEL_CACHE_DIR[$model]}"
    local src="$MODELSTORE/$cache_dir"
    local dst="$HF_CACHE/$cache_dir"
    local tmp_dst="$HF_CACHE/.pull-tmp-$cache_dir"

    if [ -d "$dst" ] && [ "$(ls -A "$dst" 2>/dev/null)" ]; then
        log "Already in local cache: $cache_dir"
        return
    fi

    if [ ! -d "$src" ] || [ -z "$(ls -A "$src" 2>/dev/null)" ]; then
        log "Not in modelstore — sparkrun will download from HuggingFace on first load."
        return
    fi

    local size
    size=$(du -sh "$src" 2>/dev/null | cut -f1)
    log "Pulling $cache_dir from modelstore ($size)..."

    # Stage into a temp dir on the same filesystem as $dst, then atomically rename
    # into place only after rsync fully succeeds. If this is interrupted (host
    # reboot, kill, network drop), $dst is never created/touched, so the
    # "already cached" check above can't be fooled by a partial transfer — the
    # same root-cause class as the 2026-08-03 ComfyUI cache-corruption incident,
    # here for model weights instead of Python packages (see docs/HISTORY.md).
    rm -rf "$tmp_dst"
    mkdir -p "$tmp_dst"
    rsync -a --info=progress2 "$src/" "$tmp_dst/"
    mv "$tmp_dst" "$dst"
    log "Pull complete."
}

# Unload whatever ComfyUI holds (FLUX, CosXL, ACE Step) before the LLM starts. A resident image or
# music model eats the LLM's reservation: on 2026-10-24 SGLang refused to start ("Raise
# --mem-fraction-static above 0.783") with FLUX still loaded. Every switch path (CLI, dashboard,
# chat tool, watchdog restart) comes through here; running a start-*.sh directly does not.
# ComfyUI applies /free between jobs, so wait (bounded) for the queue to drain, then free again.
free_comfyui() {
    local url="http://127.0.0.1:8188"
    if ! curl -sf -m 10 -X POST "$url/free" -H 'Content-Type: application/json' \
            -d '{"unload_models":true,"free_memory":true}' >/dev/null; then
        log "ComfyUI not reachable; skipping its unload."
        return 0
    fi
    local i
    for i in $(seq 1 30); do
        if curl -sf -m 5 "$url/queue" | python3 -c 'import json,sys; q=json.load(sys.stdin); sys.exit(1 if q["queue_running"] or q["queue_pending"] else 0)' 2>/dev/null; then
            # free again: a queued job that ran after the first /free would have reloaded its model
            [ "$i" -gt 1 ] && curl -sf -m 10 -X POST "$url/free" -H 'Content-Type: application/json' \
                -d '{"unload_models":true,"free_memory":true}' >/dev/null
            log "ComfyUI models unloaded."
            return 0
        fi
        [ "$i" -eq 1 ] && log "ComfyUI is busy; waiting up to 60 s for its queue to drain and unload..."
        sleep 2
    done
    log "WARNING: ComfyUI still busy after 60 s; continuing (the new model may fail to fit)."
}

[ $# -eq 1 ] || usage

MODEL="$1"
case "$MODEL" in
    minimax|qwen3.5|gpt-oss|ds4flash|qwen38flash|qwen38atlas|command-a) ;;
    *) echo "Unknown model: $MODEL"; usage ;;
esac

echo "Switching to: $MODEL"
echo ""

pull_from_modelstore "$MODEL"
free_comfyui
"$SCRIPTS_DIR/stop-model.sh"

echo "Tip: once the new model is confirmed working, run scripts/park-models.sh (dry run) then"
echo "     scripts/park-models.sh --apply to back weights up to the USB modelstore and free the"
echo "     local cache of the models that are no longer loaded (manual by design)."
echo ""
# exec, so the start script is the only holder of the lock (fd 200); it releases the lock once the
# model reports ready, and then keeps running for the model's lifetime.
exec "$SCRIPTS_DIR/${MODEL_SCRIPT[$MODEL]}"
