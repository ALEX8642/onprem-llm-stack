# onprem-llm-stack

A private LLM platform on two ASUS Ascent GX10 nodes (NVIDIA GB10 Grace Blackwell, 128 GB unified
memory each), run as one tensor-parallel inference cluster for a household and for coding agents.
Models are served TP=2 across both nodes over a 200GbE direct link (vLLM, SGLang and NVIDIA's Atlas
engine; NVFP4/FP8/MXFP4 checkpoints) behind a LiteLLM gateway. OpenWebUI sits in front with
Authentik OIDC single sign-on. Around that: an evaluation harness that gates every engine or recipe
change, a watchdog for silent model faults, speech in and out, document ingestion, nightly
disaster-recovery backups, and an Ansible rebuild of both nodes.

This is a sanitized public copy of a private working repo (fresh history; IPs, keys, domains and
usernames removed). Hostnames are `head`, `worker` and `nas`.

## Architecture

```
OpenCode / API clients             OpenWebUI :3001 (household, Authentik SSO)
        │                          │  filters/tools: vision router · deep research · context trimmer
        └──────────────┬───────────┘
                       │
                 LiteLLM :4000  (head)   — no fallbacks: an unloaded model fails loudly
   (MiniMax-M2.7 / Qwen3.5-122B / DeepSeek-V4-Flash / Qwen3.8-Flash-Next)
                       │
      ┌─────────┬──────┴───────┬──────────────────┐
  minimax :8002 qwen3 :8003  ds4flash :8000   qwen38atlas :8010
  MiniMax M2.7  Qwen3.5-122B  DeepSeek-V4-     Qwen3.8-Flash-Next
   NVFP4 (TP=2)  FP8 (TP=2)   Flash (TP=2)     NVFP4 (Atlas, TP=2)
      └─────────┴──────────────┴──────────────────┘
                [one at a time]

head                                 worker                           nas
  model rank 0                         model rank 1                     Ollama :11434 (vision)
  LiteLLM :4000                        Chatterbox TTS :8004             Whisper STT (OpenVINO,
  OpenWebUI :3001                      tts-proxy :8006                    Intel Arc), Wyoming :10300
  stt-bridge :8005 ──────────────────────────────────────────────→
  Docling + orient-fix :5001           Wyoming TTS :10300 (Home Assistant)
  ComfyUI :8188                        SearXNG :8888
  google-mcp (loopback only)
  ─────────────────── NCCL over the 200GbE CX7 link ───────────────────
```

| Path | What |
|---|---|
| `docker-compose.yml`, `worker-compose.yml` | Services on each node (critical images pinned, GPU via CDI) |
| `litellm-config.yaml`, `litellm_hooks.py` | Model aliases, deliberate no-fallback routing, request hooks |
| `scripts/switch-model.sh`, `start-*.sh`, `stop-model.sh`, `recipes/` | One-at-a-time model lifecycle across both nodes (flock against concurrent switches) |
| `scripts/bench-llm.py`, `bench-results/` | Evaluation harness and recorded runs |
| `scripts/model-watchdog.py`, `systemd/` | Silent-fault canary on a 15-minute timer |
| `scripts/smoke-test.sh` | Whole-stack check after reboots and upgrades |
| `scripts/backup.sh`, `docs/RECOVERY.md` | Nightly DR backup and the rebuild runbook |
| `docs/HISTORY.md` | Incident log: why things are shaped the way they are |
| `ansible/` | Rebuild of both nodes (NFS, monitoring/swap/earlyoom, Docker stack, model scripts) |
| `stt-bridge/`, `tts-proxy/`, `docling-orient-fix/` | Small services written for this stack (below) |

## Evaluation

`scripts/bench-llm.py` is a backend-agnostic A/B harness for any OpenAI-compatible endpoint
(standard library only). It exists so that a faster engine or quantization recipe can't hide a
quality regression. Every candidate runs the same prompts, and a change ships only if speed improves
**and** the correctness checks still pass.

| Check | What it measures |
|---|---|
| speed | Single-stream decode tok/s and TTFT over 8 task categories (prose, summary, coding, math, JSON, HTML, …), thinking off and on |
| concurrent | Aggregate and per-stream tok/s and worst-case TTFT at x2 and x4 |
| multiturn | TTFT of a follow-up turn on a ~22K-token prefix (is prefix caching actually working?) |
| needle | ~93K-token prompt: needle retrieval and prefill rate |
| tools | Tool call parsed into `tool_calls`, not leaked into `content` |
| reasoning | Thinking goes to `reasoning_content`, no `<think>` tags in `content` |
| quiz | 12 auto-graded questions with thinking on |
| vision | Whether the endpoint accepts an image at all |

Recorded runs, Qwen3.8-Flash-Next-NVFP4, TP=2 across both nodes (`bench-results/`):

| Run | Decode tok/s (median, think off / on) | x4 aggregate tok/s | x4 max TTFT | Prefix reuse (turn-2 TTFT) | 93K needle (prefill tok/s) | Tools / reasoning / quiz |
|---|---|---|---|---|---|---|
| 2026-09-26 SGLang baseline | 53.8 / 56.5 | 97.5 | 0.92 s | 7.85 → 0.51 s (**15.5×**) | pass (3,636) | pass / pass / 12/12 |
| 2026-09-26 SGLang `postfw` | 60.2 / 58.3 | 99.0 | 0.47 s | 7.48 → 0.31 s (23.9×) | **fail**: answered `!!!!…` (4,120) | pass / pass / 12/12 |
| 2026-10-24 SGLang | 56.7 / 52.4 | 101.9 | 0.42 s | 7.65 → 0.29 s (26.3×) | pass (3,654) | pass / pass / 12/12 |
| 2026-10-24 Atlas, gpu-mem 0.60 | **71.9 / 73.5** | **105.4** | **0.23 s** | 6.22 → 0.24 s (26.0×) | pass (3,989) | pass / pass / 12/12 |

The second row shows what the harness is for. That run was faster on every speed metric, yet it failed
the long-context needle with a string of `!`. That is the same NaN-logits signature the watchdog below
now guards against. The Atlas engine was adopted on 2026-10-24 because it won on speed *and* passed
everything. It also needs about 20 GiB less RAM per node: its large embedding table pages from NVMe,
and that freed room for image generation alongside the model. SGLang stays as a one-command rollback.
Memory traces taken during these runs are in the `*-mem.*` files.

## Reliability: the silent NaN-logits fault

**Detect.** On 2026-10-11 the model server was found answering short health checks correctly. But any
prompt that crossed the 4,096-token prefill chunk with thinking off came back as endless `!`
(token 0, i.e. NaN logits). A "Reply with: ok" smoke test stayed green the whole time.
`scripts/model-watchdog.py` runs every 15 minutes from a systemd timer and sends a ~6.5K-token canary
straight to the engine. **A random nonce goes first in the prompt**, so the prefix cache can never
serve it. A cached canary is processed as one small chunk and would pass while the model was broken.

**Contain.** On a bad answer the watchdog saves evidence before touching anything: the canary result,
engine metrics and server info, both nodes' kernel logs, GPU visibility inside each model container,
memory, and both ranks' serve logs. It skips the check while a model switch holds the switch lock, so it never fights a
deliberate change.

**Correct.** The ladder runs flush the cache → re-test → restart the model through the model-control
API (shared lock with `switch-model.sh`). Restarts are limited to once per 6 hours.

**Escalate.** If the model is still broken after a restart, the watchdog stops. It writes
`NEEDS_ATTENTION` and hands off to a person with the manual fix in the email. Mail goes out at every
step (fault detected, fixed by flush, restart triggered, needs a person, recovered with downtime, or
"not back after 25 minutes"). Handled states exit 0, so only a crash of the watchdog itself trips
systemd's `OnFailure` alert.

## Operations

- **Rebuild.** `ansible-playbook -i ansible/hosts.yml ansible/site.yml` (tags: `nfs`, `monitoring`,
  `docker`, `scripts`). `docs/RECOVERY.md` is the bare-metal order of operations, from OS reinstall
  through interconnect, CDI, sparkrun, images and data to verification with `bench-llm.py` against
  `bench-results/`.
- **DR backup.** `scripts/backup.sh` runs nightly on a systemd timer. It takes hard-linked snapshots
  (14 kept) of OpenWebUI data, secrets, `/etc` and version manifests from both nodes, local patches to
  upstream checkouts, and saved images. Every section is independent, so a partial failure is a warning
  and the rest still lands. Copies are integrity-checked (SQLite `integrity_check`, `tar -t`,
  `zstd -t`, git bundle verify). A destination is only written if it carries a marker file, so an
  unmounted drive can't silently fill the root disk. The OpenWebUI restore was tested into a throwaway
  container.
- **Version pinning, and why.** The stateful and API-critical images (LiteLLM, OpenWebUI, Docling)
  are pinned by digest or version and bumped deliberately, with the test to run after each one
  written next to the pin. On the hosts:
  - **GPU driver held on the 580 series.** 590/595 caused kernel panics and deadlocks on GB10.
  - **Kernel held.** NVIDIA's advisory says the 7.0 line breaks multi-node NCCL over RoCE.
  - **No blanket `apt upgrade`.** Changes go to the worker first as a canary.
- **CDI migration.** GPU containers silently lost their devices
  (`Failed to initialize NVML: Unknown Error`) after any `systemctl daemon-reload`. snapd triggers
  those. The cause was the legacy `driver: nvidia` hook under cgroup v2. Every GPU service now
  requests CDI devices, and the fix was proven with a deliberate `daemon-reload` on each node
  (`docs/HISTORY.md`).
- **Routing.** LiteLLM has no fallbacks on purpose. With one model loaded at a time, a fallback would
  answer from a different model without telling anyone.
- **Least privilege for the Google MCP server.** Gmail, Calendar and Drive are reachable only from
  OpenWebUI's admin connection. The server sits on its own Docker network shared only with OpenWebUI,
  its host port binds to loopback for the one-time sign-in, and it runs with drafts-only Gmail scope,
  send and label tools disabled, and read-only Calendar and Drive.

### Small services

- **`stt-bridge/`** turns OpenAI `/v1/audio/transcriptions` into a Whisper backend on the NAS (OpenVINO on
  an Intel Arc GPU, with a Wyoming fallback). It replaced a GPU Whisper container that never worked on
  arm64, because PyPI's CTranslate2 wheel has no CUDA there.
- **`tts-proxy/`** is an OpenAI `/v1/audio/speech` front for Chatterbox TTS, adding emotion control.
- **`docling-orient-fix/`** is a temporary proxy that rotates scanned pages upright before Docling
  sees them. It works around upstream bug
  [docling#3839](https://github.com/docling-project/docling/issues/3839), which was filed from this
  stack.

## How this was built

Alexander Feht set the architecture, constraints and acceptance criteria. He ran the benchmarks and
incident investigations and operates the cluster. Most of the code was written with AI coding agents
(Claude Code, Codex, GitHub Copilot CLI) under his specification and review.

Not published here: secrets (`.env`), SearXNG settings, OpenWebUI functions other than
`auto_thinking.py`, the model-control API, ComfyUI workflows, UI assets, and the private notes that
some comments still reference (STATUS, MODELS, TROUBLESHOOTING).
