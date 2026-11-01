#!/usr/bin/env bash
# Whole-stack smoke test. Read-only apart from one chat completion, one embedding and one
# TTS->STT round-trip. Run after a reboot, an image bump or a config change.
#   bash scripts/smoke-test.sh            # MODEL defaults to the Qwen3.8 trial's fast alias
#   MODEL=MiniMax-M2.7-Best bash scripts/smoke-test.sh
# Exit status = number of failed checks.
set -u
WORKER="${WORKER:-worker}"
HEAD="${HEAD:-head}"
MODEL="${MODEL:-Qwen3.8-Flash-Next-NVFP4-Fast}"
KEY="${LITELLM_KEY:?set LITELLM_KEY (the LiteLLM master key)}"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAIL=0
ok()   { printf '  \033[32mPASS\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
chk()  { local name="$1"; shift; if out=$("$@" 2>&1); then ok "$name ${out:+— $out}"; else bad "$name ${out:+— $out}"; fi; }
w()    { ssh -o ConnectTimeout=8 -o BatchMode=yes "$WORKER" "$@"; }
http() { curl -sf -m "${2:-10}" -o /dev/null -w '%{http_code}' "$1"; }

echo "== host"
chk "head swap"  bash -c '[ -n "$(swapon --noheadings)" ] && swapon --noheadings --show=SIZE,USED | tr -s " " | xargs'
chk "worker swap"  bash -c "[ -n \"\$(ssh -o BatchMode=yes $WORKER swapon --noheadings)\" ] && echo on"
chk "head CDI spec" test -s /etc/cdi/nvidia.yaml
chk "worker CDI spec" w test -s /etc/cdi/nvidia.yaml

echo "== GPU access inside containers (the NVML-after-daemon-reload failure mode)"
for c in comfyui ollama-local; do chk "head $c" bash -c "docker exec $c nvidia-smi -L | head -1 | cut -c1-20"; done
for c in chatterbox; do chk "worker $c" bash -c "ssh -o BatchMode=yes $WORKER docker exec $c nvidia-smi -L | head -1 | cut -c1-20"; done

echo "== endpoints"
chk "open-webui"        bash -c "curl -sf -m 10 http://$HEAD:3001/api/version"
chk "litellm"           http "http://$HEAD:4000/health/liveliness"
chk "docling-orient-fix" http "http://$HEAD:5001/health"
chk "comfyui"           http "http://$HEAD:8188/system_stats"
chk "model-control"     http "http://$HEAD:9091/"
chk "searxng"           http "http://$WORKER:8888/search?q=test&format=json" 15
chk "nas ollama"     http "http://nas:11434/api/tags"

echo "== inference"
chk "chat ($MODEL)" bash -c "curl -sf -m 120 http://$HEAD:4000/v1/chat/completions -H 'Authorization: Bearer $KEY' \
  -H 'Content-Type: application/json' -d '{\"model\":\"$MODEL\",\"max_tokens\":20,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with: ok\"}]}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"choices\"][0][\"message\"][\"content\"].strip()[:40])'"
# A ~6K-token prompt: crosses SGLang's 4,096-token prefill chunk. On 2026-10-11 the Qwen3.8 server
# answered short prompts fine but returned endless "!!!!" (token 0) for anything over ~4K tokens.
# The random prefix keeps the prefix cache from serving it (cached = one small chunk = false pass).
chk "long prompt (~6K tokens, no '!!!!')" bash -c "python3 - <<'PY'
import json,sys,urllib.request
text=open('$(dirname "$0")/../docs/MODELS.md').read()*3
import secrets
body={'model':'$MODEL','max_tokens':30,'messages':[{'role':'user','content':'[smoke '+secrets.token_hex(8)+']\\n'+text[:16000]+'\\n\\nIn one sentence, what is this document about?'}]}
r=urllib.request.Request('http://$HEAD:4000/v1/chat/completions',json.dumps(body).encode(),{'Authorization':'Bearer $KEY','Content-Type':'application/json'})
d=json.load(urllib.request.urlopen(r,timeout=180)); c=d['choices'][0]['message'].get('content') or ''
print(d['usage']['prompt_tokens'],'tok:',repr(c[:40])); sys.exit(1 if '!!!' in c or not c.strip() else 0)
PY"
chk "embeddings" bash -c "curl -sf -m 60 http://$HEAD:4000/v1/embeddings -H 'Authorization: Bearer $KEY' -H 'Content-Type: application/json' \
  -d '{\"model\":\"nomic-embed-text:v1.5\",\"input\":\"hello\"}' | python3 -c 'import json,sys; print(len(json.load(sys.stdin)[\"data\"][0][\"embedding\"]),\"dims\")'"
chk "TTS (tts-proxy → chatterbox)" bash -c "curl -sf -m 120 -o $TMP/t.mp3 http://$WORKER:8006/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{\"model\":\"tts-1\",\"voice\":\"Abigail.wav\",\"input\":\"Smoke test passed.\"}' && echo \$(stat -c %s $TMP/t.mp3) bytes"
chk "STT bridge → NAS Wyoming whisper" bash -c "curl -sf -m 10 http://$HEAD:8005/health | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d[\"backend\"], d[\"asr\"])'"
chk "STT (transcribes the TTS clip)" bash -c "curl -sf -m 120 http://$HEAD:8005/v1/audio/transcriptions -F file=@$TMP/t.mp3 -F model=whisper-1 \
  | python3 -c 'import json,sys; t=json.load(sys.stdin)[\"text\"]; print(t); sys.exit(0 if \"smoke\" in t.lower() else 1)'"
chk "google-mcp (Google token valid)" bash "$(dirname "$0")/google-mcp.sh" check

echo "== memory"
for h in localhost "$WORKER"; do
  ssh -o BatchMode=yes "$h" "free -g | awk '/Mem:/{printf \"  %-14s avail %sG of %sG\", \"$h\", \$7, \$2} /Swap:/{printf \", swap used %sG\\n\", \$3}'"
done

echo; [ "$FAIL" -eq 0 ] && echo "ALL PASS" || echo "$FAIL check(s) FAILED"
exit "$FAIL"
