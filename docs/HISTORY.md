# History — resolved incidents and why things are shaped this way

> One short entry per incident: what happened, the fix, and the lesson that still matters.
> Scrubbed copy: some referenced files (STATUS/MODELS/TROUBLESHOOTING notes, OpenWebUI functions)
> are kept private.

## 2026-10-11 cleanup pass

- **GPU access silently lost after systemd reloads.** Chatterbox (TTS, Home Assistant voice) and
  Whisper on worker had `Failed to initialize NVML: Unknown Error` after snapd-triggered
  `daemon-reload`s on Oct 5/8. ollama-local on head hit the same thing in September. Cause: the
  legacy `driver: nvidia` device hook under cgroup v2 + systemd. Fix: every GPU service in both
  compose files now requests CDI devices (`driver: cdi`, `nvidia.com/gpu=all`); worker got its
  first `/etc/cdi/nvidia.yaml`. Proven with a deliberate `daemon-reload` on each node. The old note
  "CDI spec is required" was only half the fix, because nothing actually requested CDI devices.
- **Speech-to-text had been broken since 2026-06-08.** It worked until then on OpenWebUI's built-in
  CPU Whisper (`base`). The GPU faster-whisper container on worker that replaced it never loaded a
  model: on arm64 PyPI's CTranslate2 has no CUDA (16.9 MB wheel vs 39.6 MB on x86), and its old
  faster_whisper rejected the `large-v3-turbo` shorthand. A GPU rebuild was rejected (fragile, and it
  costs RAM on the tight unified pool). Fix: the `stt-bridge` container on head translates OpenAI
  `/v1/audio/transcriptions` to Wyoming and forwards to the faster-whisper that already runs on NAS
  for Home Assistant (~0.5 s per clip). worker's Whisper was retired, freeing ~3.4 GB.
- **head ran without swap for two weeks.** Swap went off during the 2026-09-28 fieldiag prep
  session and nothing turned it back on. `swapon -a` restored it. After any
  maintenance, check `swapon --show`. (worker had separately lost swap on every reboot until
  2026-09-26, because an old Ansible run had masked `swap.img.swap`.)
- **The "OpenWebUI Models page bug" was a misdiagnosis.** The 2026-08-28 live patch to
  `Models.search_models()` targeted Workspace → Models, which lists custom models only by design.
  Admin → Models never called it. Retired (TROUBLESHOOTING.md).
- **OpenWebUI 0.11.4 → 0.12.0**, now pinned. The dead Ollama (`host.docker.internal`) and keyless
  api.openai.com connections were removed. LiteLLM and docling were pinned by digest.

## OpenWebUI

- **`WEBUI_SECRET_KEY` must be set explicitly** (2026-08-03). Without it, every container recreation
  generated a new key and logged everyone out. It lives in `.env`.
- **Image Generator** (2026-07-01): a plain-HTTP ComfyUI link is mixed content on the HTTPS site;
  `data:` URIs get routed to the model, not the user. The tool uploads through OpenWebUI's own
  file storage and attaches via a `files` event.
- **Auto Vision Router** (2026-08-03): minicpm-v can't decode WebP, so every image is re-encoded to
  PNG first.
- **Auto Memory** (third-party, DB-stored, 2026-08-28): patched a missing `await`
  (`get_user_by_id` went async in 0.11.1), and its valves were repointed from api.openai.com to
  LiteLLM. Both patches survive image updates. Re-check after a bump.
- **v0.9.6 retrieval regression** (removed 2026-08-03): the `BYPASS_RETRIEVAL_ACCESS_CONTROL`
  workaround was dropped once upstream PR #25600 shipped. It bypassed KB access control broadly, so
  it must not come back.
- **TTS voice list** shows generic names: tts-proxy's `/v1/audio/voices` returns plain strings, not
  `{id,name}` objects. Cosmetic.

## ComfyUI

- **xformers** (2026-07-01): the GB10 dev build lacks `memory_efficient_attention`, hence
  `--use-pytorch-cross-attention`.
- **Corrupted `uv_cache` after a host reboot mid-write** (2026-08-03): zero-byte METADATA/WHEEL files
  crash-looped the container. `comfyui-preflight.sh` (entrypoint override) now detects and clears
  that signature on every start.

## Scripts and robustness (2026-08-03 hardening pass)

- `switch-model.sh` pulls models from the modelstore into a temp dir and only `mv`s into place after
  rsync succeeds. Before that, a partial cache entry could fool it into re-downloading 155 GB from
  HF.
- Start scripts trap TERM/INT/HUP (a bare `trap EXIT` didn't fire while blocked in the polling
  loop) and run `sparkrun stop --all`, so an interrupted launch doesn't orphan worker's shard. That
  can't cover a reboot or SIGKILL.
- litellm and docling have healthchecks, and `docling-orient-fix` waits on docling's health. That
  only applies to `docker compose up`, not to the daemon's own restarts.
- LiteLLM has **no `fallbacks`** (2026-07-01). With one model up at a time, fallbacks silently
  answered from the wrong model.

## Models

- **DeepSeek-V4-Flash-0731 trial** (2026-08-07/08). Boot needed:
  - `sparkrun tune vllm` once for the FlashInfer sparse-MLA autotune (~3.5 h, cached by
    `ensure_tuning_cache()`).
  - gpu-mem 0.80, so the ~35 s DeepGEMM warmup spike doesn't push worker into earlyoom. 0.80 is
    reverse-engineered from the 2× concurrency target, not "more headroom".
  - A derived image adding `ray` and clearing the `ENTRYPOINT` that swallowed sparkrun's Ray
    bootstrap.
  - `hosted_vllm/` in LiteLLM so `reasoning_effort` reaches vLLM. `openai/` 400s, and
    `allowed_openai_params`/`extra_body` silently drop it. LiteLLM reports the field as
    `reasoning_content`.
- **Swap re-added on worker** (2026-08-08, reversing commit `77afd0d`): the user prefers slower under
  pressure over crashed. Docker's `--memory-swappiness` is ignored on cgroup v2, so per-container
  swap tuning isn't an option. earlyoom (`-m 2 -s 80`, conjunctive) stays as the backstop.
- **Declined optimizations** (2026-07-21): `--enforce-eager` on MiniMax, because CUDA graphs are
  noticeably faster. vLLM sleep mode for FLUX pauses is untested; see STATUS "Not Yet Done" and
  vLLM #50011.
- **Disk:** head's root filled to 100% mid-pull (2026-08-28) from 78 GB of stale Docker build
  cache. Run `docker builder prune -f` now and then.

## Upstream bugs we filed or track

- [docling#3839](https://github.com/docling-project/docling/issues/3839) (ours, open): text from
  rotated scans comes out scrambled. Hence `docling-orient-fix`.
- OpenWebUI KB-name search uses whole-query `ilike` (open, not filed). Use the `*_knowledge_files`
  tools instead.
