# Disaster Recovery — rebuilding head/worker from scratch

Written 2026-09-26. Goal: if both Sparks were wiped tomorrow, get back to the same working stack
using only (a) this GitHub repo, (b) the USB modelstore drive, (c) the external services listed
below. Everything the repo doesn't hold is captured by `scripts/backup.sh`.

## What lives where

| What | Where | Refreshed by |
|---|---|---|
| Stack config, scripts, recipes, docs, functions, workflows | this git repo + a `git bundle` in every backup snapshot | `git push` (manual) / nightly |
| OpenWebUI (users, chats, functions, knowledge, uploads, vector DB) | `<dest>/state/<ts>/openwebui/` | nightly timer |
| Secrets: `.env`, SSH keys (both nodes), `~/.config/sparkrun` | `<dest>/state/<ts>/secrets/` | nightly timer |
| All of `/etc` (netplan/CX7 IPs, fstab, sudoers.d, systemd units, earlyoom, docker daemon.json, samba, CDI) per node | `<dest>/state/<ts>/system/{head,worker}/etc.tar.gz` | nightly timer |
| Version manifest per node (DGX OS, kernel, driver, firmware, apt holds, image digests, sparkrun) + `dpkg --get-selections` | `<dest>/state/<ts>/system/{head,worker}/` | nightly timer |
| worker's deployed stack files (worker has **no git checkout** — files are copied over) | `<dest>/state/<ts>/worker/ai-stack/` | nightly timer |
| Chatterbox voices, reference audio, config | `<dest>/state/<ts>/worker/chatterbox/` | nightly timer |
| Local patches to Chatterbox-TTS-Server / ComfyUI-Nvidia-Docker checkouts | `<dest>/state/<ts>/local-mods/*.diff` (+ upstream HEAD) | nightly timer |
| ComfyUI outputs, custom nodes, user settings, model manifest | `<dest>/state/<ts>/comfyui/` | nightly timer |
| ComfyUI model files (37 GB; CosXL is gated) | `<dest>/comfyui-models/` (mirror) | nightly timer (`--all`) |
| Docker images that can't be reliably re-pulled/rebuilt | `<dest>/images/*.tar.zst` + `.id` + `.digest` | nightly timer (`--all`; skips unchanged) |
| LLM weights | USB `/mnt/modelstore/huggingface/hub/` (all re-downloadable from HF too) | `scripts/park-models.sh` (manual) |

`<dest>` = each entry in `BACKUP_DESTS` (colon-separated; set in `.env`). Default and currently the
only one: **`/mnt/modelstore/backups/ai-stack`** on the USB drive attached to head.

**Known gap — no off-box copy yet.** The USB drive survives an OS wipe but not head hardware
loss/theft/drive failure. To close it: export a dedicated NAS share to head,
mount it with `nofail`, and add it: `BACKUP_DESTS=/mnt/modelstore/backups/ai-stack:/mnt/<nas-mount>/ai-stack`.
The state tier is ~300 MB so it's cheap to send off-box; images/ComfyUI models are optional there.

**External dependencies (not on the cluster, must still exist):** the Authentik OIDC provider
(OpenWebUI OAuth app, client id/secret in `.env`), split-horizon LAN DNS so the SSO hostname
resolves internally, the NAS (Ollama vision model, Whisper STT), the reverse proxy in front of
OpenWebUI, and a Hugging Face token (gated CosXL license accepted).

## Backup operations

```bash
systemctl list-timers ai-stack-backup.timer        # nightly ~04:00, Persistent=true
journalctl -u ai-stack-backup.service -n 50        # last run log
cat /mnt/modelstore/backups/ai-stack/LAST_RUN.txt  # status OK / WARNINGS + sizes
scripts/backup.sh                                  # state tier only, ad hoc (~10 s)
scripts/backup.sh --all                            # + images + ComfyUI models
```

State snapshots are hard-linked against the previous one (unchanged files cost nothing); the newest
14 are kept (`KEEP=`). Each run checks the OpenWebUI and Chroma DB copies (both taken with SQLite's
online-backup API; `PRAGMA integrity_check` + row counts → `openwebui/CHECK.txt`), `tar -t`s the
/etc archives, verifies the git bundle, and `zstd -t`s each image archive.

- **Destination marker:** a destination is only written if it contains `.ai-stack-backup-dest`
  (created once by hand on the real drive/share). An unmounted USB drive or stale NFS mount is
  skipped with a warning rather than silently filling the root disk under the empty mountpoint.
  New destination → `touch <dest>/.ai-stack-backup-dest`.
- **Partial failure:** every section is independent — if worker or a container is down, that
  section is a WARN, the rest still lands, `LAST_RUN.txt` says WARNINGS, and the unit shows failed.
  A failed run triggers `ai-stack-notify@` (OnFailure) → email via `scripts/notify-failure.py` once
  SMTP settings are in `.env` (until then it only logs). Also: `LAST_RUN.txt` / `systemctl status ai-stack-backup`.
- The ComfyUI models mirror never deletes (and refuses to sync if the source shrank below half the
  backup), so an emptied `~/comfyui/models` can't wipe the backup copy.
- Image archives keep one version each (latest save); a broken re-pull of a moving tag replaces it. Unit files live in
`systemd/`; if they change: `sudo cp systemd/ai-stack-backup.* systemd/ai-stack-model-watchdog.* systemd/ai-stack-notify@.service /etc/systemd/system/ && sudo systemctl daemon-reload`.

**Restore-tested (2026-09-26):** OpenWebUI — the snapshot's `webui.db` + `data/` were loaded into a
throwaway container of the current image on :3999: healthy, all users/chats/files/functions present, no errors (same procedure as step 8 below).
Chroma vector store: integrity-checked (collection and embedding counts matched), not query-tested on the
restored instance — if knowledge search misbehaves after a restore, re-index from the restored uploads.
**Not yet tested:** a full bare-metal rebuild; `docker load` of the image archives (archives are
verified with `zstd -t`, the load itself isn't).

## Restore runbook (in order)

Pick the newest snapshot: `B=$(ls -1d /mnt/modelstore/backups/ai-stack/state/20* | tail -1)`.
Its `system/{head,worker}/MANIFEST.txt` says exactly which versions were running.

1. **Reinstall DGX OS** on both nodes (NVIDIA recovery media). Match the manifest: as of
   2026-09-26 that's DGX OS 7.5.0, kernel **6.17.0-1021-nvidia**, driver **580.159.03**.
   Do **not** take kernel 7.0.0-1019 (NVIDIA advisory: breaks multi-node NCCL/RoCE; also reserves
   ~7 GiB RAM via KHO) or any 590+/595 driver — see "Version pinning" in the README.
   Consider `sudo apt-mark hold linux-image-nvidia-hwe-24.04 linux-headers-nvidia-hwe-24.04`.
   Hostnames head/worker, user `ai`.
2. **Mount the USB drive on head** at `/mnt/modelstore` (fstab line is in
   `system/head/etc.tar.gz` → `etc/fstab`; `nofail`).
3. **Secrets + repo.** `mkdir -p ~/.ssh && rsync -a $B/secrets/ssh-head/ ~/.ssh/` (worker: `ssh-worker/`), fix perms
   (`chmod 700 ~/.ssh; chmod 600 ~/.ssh/id_*`). `git clone <this repo> ~/ai-stack`
   (offline: `git clone $B/repo/ai-stack.bundle ~/ai-stack`). `install -m 600 $B/secrets/env ~/ai-stack/.env`.
4. **CX7 interconnect.** Extract `etc/netplan/*` (and `etc/NetworkManager/system-connections/`)
   from each node's `etc.tar.gz`, `sudo netplan apply`. Expect head `enp1s0f0np0 head-cx7`,
   worker `worker-cx7` (second port pair <cx7-subnet-1>). Verify `ping worker-cx7`, `ibdev2netdev`.
5. **Base system (both nodes).** Docker + NVIDIA container toolkit ship with DGX OS; then
   `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`. Restore from `etc.tar.gz`:
   `etc/sudoers.d/{ai-stack-model-control,sparkrun-*}`, `etc/docker/daemon.json`,
   `etc/default/earlyoom`, `etc/systemd/system/{spark-tune,restart-resolved}.service` (compare
   before copying — don't blindly overwrite a fresh /etc). Swap + earlyoom:
   `ansible-playbook -i ansible/hosts.yml ansible/site.yml --tags monitoring` (fixed 2026-09-26 —
   it used to *disable* swap). The `nfs` tag targets an export that has gone stale — check first.
6. **sparkrun.** `uv tool install sparkrun==<version from manifest>` (0.3.10 as of 2026-09-26), then
   `rsync -a $B/secrets/sparkrun-config/ ~/.config/sparkrun/` (cluster `muthur` on <cx7-subnet-0>),
   `sparkrun update`.
7. **Docker images.** For each archive: `zstd -dc <dest>/images/X.tar.zst | docker load`.
   head needs all except `ai-stack-tts-proxy` and `chatterbox-tts-server`; worker needs `chatterbox-tts-server:cu130`,
   `ai-stack-tts-proxy` (or rebuild: `docker compose -f worker-compose.yml build tts-proxy`) — both archives are saved from worker; head has an older, unused chatterbox build —  plus
   whatever model image is active (sparkrun distributes model images to worker itself).
   Everything else pulls from registries; `.digest` files / MANIFEST pin exact digests if a tag moved.
8. **OpenWebUI data** (restore-tested procedure):
   ```bash
   docker volume create open-webui
   docker run --rm -v open-webui:/d -v $B/openwebui:/b:ro alpine \
     sh -c 'cp -a /b/data/. /d/ && cp /b/webui.db /d/webui.db'
   ```
   `WEBUI_SECRET_KEY` must be the one from the backed-up `.env`, or every session/OAuth token is invalid.
9. **head stack.** `cd ~/ai-stack && docker compose up -d` (builds `docling-orient-fix`).
   Pull the embedding model: see the header comment in `docker-compose.yml`.
10. **worker stack.** Copy `worker-compose.yml`, `searxng/`, `tts-proxy/` from the repo (or
    `$B/worker/ai-stack/`) to worker `~/ai-stack/`, plus `.env` (`$B/secrets/env-worker`).
    Chatterbox: `git clone <url in local-mods/Chatterbox-TTS-Server.HEAD>`, checkout that commit,
    `git apply $B/local-mods/Chatterbox-TTS-Server.worker.diff`, restore
    `$B/worker/chatterbox/{config.yaml,voices,reference_audio}`. Then
    `docker compose -f worker-compose.yml up -d`.
11. **ComfyUI.** Clone ComfyUI-Nvidia-Docker at the recorded HEAD, `git apply` its diff, then
    `rsync -a <dest>/comfyui-models/ ~/comfyui/models/`, restore `comfyui/{output,custom_nodes}`
    and `comfyui/user/` → `~/ComfyUI-Nvidia-Docker/run/ComfyUI/user/`. `docker compose up -d comfyui`
    (first start rebuilds its venv — slow, expected).
12. **Host services.** `ansible-playbook ... --tags scripts` (model-control API unit + sudoers),
    then the backup timer: `sudo cp systemd/ai-stack-backup.* systemd/ai-stack-model-watchdog.* systemd/ai-stack-notify@.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now ai-stack-backup.timer ai-stack-model-watchdog.timer`.
13. **Model.** `scripts/switch-model.sh qwen38atlas` (or whichever was active; `qwen38flash` is the
    SGLang rollback) — pulls weights from the USB modelstore, sparkrun pushes them to worker. The
    nvidia checkpoint Atlas uses reaches the modelstore only once `park-models.sh --apply` has run
    (pending as of 2026-10-24); until then a rebuild downloads it from HF (124 GiB).
14. **Verify.** SSO login at `https://chat.example.com`; `scripts/bench-llm.py` against the model and
    compare with `bench-results/`; TTS/STT from a chat; one ComfyUI generation; then run
    `scripts/backup.sh` once by hand and check `LAST_RUN.txt`.
