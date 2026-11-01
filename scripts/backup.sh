#!/usr/bin/env bash
# ai-stack disaster-recovery backup. Captures everything needed to rebuild head/worker that is NOT
# already in git (see docs/RECOVERY.md for what's where and the restore order).
#
#   backup.sh                 state tier: small, irreplaceable stuff — run nightly by
#                             ai-stack-backup.timer (systemd/ai-stack-backup.{service,timer})
#   backup.sh --images        also docker-save the images we can't reliably re-pull/rebuild
#   backup.sh --comfy-models  also mirror ~/comfyui/models (37 GB, CosXL is gated)
#   backup.sh --all           both of the above
#
# Destinations: every dir in BACKUP_DESTS (default: USB modelstore). Add an off-box target (NAS)
# via BACKUP_DESTS in .env once a dedicated share exists — the USB drive is attached to head and
# does not survive hardware loss/theft. A destination is only used if it contains the marker file
# .ai-stack-backup-dest (create it once by hand) — so an unmounted USB drive / stale NFS share is
# skipped with a warning instead of silently filling the root disk under an empty mountpoint.
# State snapshots are hard-linked against the previous one (rsync --link-dest), so unchanged files
# cost no extra space; the newest KEEP snapshots are kept.
#
# Every section is independent: if worker or a container is down, that section is logged as a
# WARN, the rest still lands in the snapshot, and the script exits 1 (the systemd unit shows failed).
# Snapshots contain secrets (.env, SSH keys, /etc) — directories are mode 700; never point
# BACKUP_DESTS at a world-readable share. Created 2026-09-26.
set -euo pipefail
umask 077

STACK="$HOME/ai-stack"
[ -f "$STACK/.env" ] && BACKUP_DESTS_ENV=$(grep -E '^BACKUP_DESTS=' "$STACK/.env" | cut -d= -f2- | tr -d '"' || true)
BACKUP_DESTS="${BACKUP_DESTS:-${BACKUP_DESTS_ENV:-/mnt/modelstore/backups/ai-stack}}"
KEEP="${KEEP:-14}"
WORKER="${WORKER:-worker}"
MARKER=.ai-stack-backup-dest
IMAGES=0; COMFY_MODELS=0
for a in "$@"; do case "$a" in
    --images) IMAGES=1 ;; --comfy-models) COMFY_MODELS=1 ;; --all) IMAGES=1; COMFY_MODELS=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;; esac; done

# Images that are locally built, or pinned upstream builds a model/service depends on (tags do vanish:
# the vLLM nightly tag was renamed 2026-09-26). Everything else is re-pullable from its registry.
# One archive per image (latest only) — a broken re-pull of a moving tag overwrites the previous one.
SAVE_IMAGES_SPARK0=(
    "ghcr.io/atlas-inf/atlas-gb10:sha-b9a7eda"          # running Qwen3.8 Atlas recipe (since 2026-10-24)
    "ghcr.io/thupalo/qwen38-flashnext-sm121:20260827"   # Qwen3.8 SGLang recipe (rollback)
    "ds4flash-fixed:0.1.1"                              # locally patched (scripts/recipes/Dockerfile.ds4flash-fix)
    "ghcr.io/anemll/dspark-vllm-gx10:0.1.1"             # base of the above
    "eugr/spark-vllm:latest"                            # MiniMax recipe; :latest moves
    "ai-stack-docling-orient-fix:latest"                # local build (docling-orient-fix/)
    "ai-stack-stt-bridge:latest"                        # local build (stt-bridge/)
    "ai-stack-google-mcp:2.1.0"                         # built from a pinned git commit (no upstream image)
    "mmartial/comfyui-nvidia-docker:ubuntu24_cuda13.1-dgx-latest"
    "open-webui-rollback:0.11.4"                        # local tag = the pre-0.12.0 image; pairs with
                                                        # archive/owui-pre-0.12/ for a rollback
)
SAVE_IMAGES_SPARK1=(
    "ai-stack-tts-proxy:latest"                         # local build (tts-proxy/)
    "chatterbox-tts-server:cu130"                       # local build, cu130 patches — worker's is the
                                                        # one running (head has a different, stale build)
)

# fail fast (not hang) if worker is down
ssh() { command ssh -o ConnectTimeout=10 -o BatchMode=yes "$@"; }
export RSYNC_RSH="ssh -o ConnectTimeout=10 -o BatchMode=yes"

TS=$(date +%Y-%m-%d_%H%M%S)
WORK=$(mktemp -d "/tmp/ai-stack-backup.XXXX")
trap 'rm -rf "$WORK"' EXIT
log() { echo "[$(date '+%H:%M:%S')] $*"; }
FAIL=0
warn() { log "WARN: $*"; FAIL=1; }
# section <name> <fn>: run fn in a subshell with errexit active; a failure is a WARN, not an abort
section() {
    local name="$1" rc; shift
    log "$name"
    set +e; ( set -e; "$@" ); rc=$?; set -e
    [ $rc = 0 ] || warn "section failed: $name (rc=$rc)"
}
S="$WORK/s"; mkdir -p "$S"/{openwebui,secrets,system/head,system/worker,worker,local-mods,comfyui,repo}

sqlite_copy_check() {  # <db file>: integrity + row counts → stdout; non-zero if not ok
    python3 - "$1" <<'EOF'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
ok = c.execute("PRAGMA integrity_check").fetchone()[0]
print("integrity_check:", ok)
for t in ("user", "chat", "knowledge", "file", "function", "model", "config"):
    try: print(f"{t}: {c.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]}")
    except Exception as e: print(f"{t}: n/a ({e})")
sys.exit(0 if ok == "ok" else 1)
EOF
}

sec_openwebui() {
    # online SQLite backup API: consistent while the app is live (a plain file copy of a WAL DB isn't)
    docker exec open-webui python3 -c "
import sqlite3
src = sqlite3.connect('/app/backend/data/webui.db'); dst = sqlite3.connect('/tmp/webui-backup.db')
src.backup(dst); dst.close(); src.close()"
    docker cp -q open-webui:/tmp/webui-backup.db "$S/openwebui/webui.db"
    docker exec open-webui rm -f /tmp/webui-backup.db
    sqlite_copy_check "$S/openwebui/webui.db" > "$S/openwebui/CHECK.txt"
    # uploads + vector store; cache/ is regenerable
    local V=/var/lib/docker/volumes/open-webui/_data
    sudo -n rsync -a --exclude=cache --exclude='webui.db*' --exclude='chroma.sqlite3*' "$V/" "$S/openwebui/data/"
    # Chroma's own SQLite gets the online-backup treatment too (its HNSW segment dirs are plain files)
    sudo -n python3 -c "
import sqlite3, sys
s = sqlite3.connect('file:$V/vector_db/chroma.sqlite3?mode=ro', uri=True); d = sqlite3.connect('$S/openwebui/data/vector_db/chroma.sqlite3')
s.backup(d); d.close(); s.close()"
    sudo -n chown -R "$(id -u):$(id -g)" "$S/openwebui/data"
    python3 -c "import sqlite3,sys; sys.exit(0 if sqlite3.connect('$S/openwebui/data/vector_db/chroma.sqlite3').execute('PRAGMA integrity_check').fetchone()[0]=='ok' else 1)"
}

sec_secrets_head() {
    cp -a "$STACK/.env" "$S/secrets/env"
    cp -a "$HOME/.ssh" "$S/secrets/ssh-head"
    rsync -a --exclude=cache "$HOME/.config/sparkrun/" "$S/secrets/sparkrun-config/"
}
sec_secrets_worker() {
    mkdir -p "$S/secrets/ssh-worker"
    ssh "$WORKER" 'tar -C ~/.ssh -cf - .' | tar -C "$S/secrets/ssh-worker" -xf -
    ssh "$WORKER" 'cat ~/ai-stack/.env' > "$S/secrets/env-worker"
}

MANIFEST_CMDS='
set +e
echo "## host";          hostname; cat /etc/dgx-release 2>/dev/null; uname -r
echo "## driver";        nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null
echo "## firmware";      cat /sys/class/dmi/id/bios_version; fwupdmgr get-devices --no-unreported-check 2>/dev/null | grep -E "^[├└│ ]*[─]|Current version" | head -40
echo "## apt holds";     apt-mark showhold
echo "## docker";        docker version --format "{{.Server.Version}}"
echo "## images (tag id digest)"; docker image ls --format "{{.Repository}}:{{.Tag}} {{.ID}} {{.Digest}}" --digests
echo "## containers";    docker ps -a --format "{{.Names}} {{.Image}} {{.Status}}"
echo "## crontab";       crontab -l 2>/dev/null
echo "## swap";          swapon --show
echo "## sparkrun";      ~/.local/share/uv/tools/sparkrun/bin/sparkrun --version 2>/dev/null
echo "## uv tools";      ~/.local/bin/uv tool list 2>/dev/null
'
sec_system_head() {
    bash -c "$MANIFEST_CMDS" > "$S/system/head/MANIFEST.txt" 2>&1
    dpkg --get-selections > "$S/system/head/dpkg-selections.txt"
    sudo -n tar -C / -czf "$S/system/head/etc.tar.gz" etc 2>/dev/null || [ $? = 1 ]  # 1 = "file changed as we read it"
    sudo -n chown "$(id -u):$(id -g)" "$S/system/head/etc.tar.gz"
    tar -tzf "$S/system/head/etc.tar.gz" >/dev/null
}
sec_system_worker() {
    ssh "$WORKER" "bash -s" <<<"$MANIFEST_CMDS" > "$S/system/worker/MANIFEST.txt" 2>&1
    ssh "$WORKER" dpkg --get-selections > "$S/system/worker/dpkg-selections.txt"
    ssh "$WORKER" 'sudo -n tar -C / -czf - etc 2>/dev/null; [ $? -le 1 ]' > "$S/system/worker/etc.tar.gz"
    tar -tzf "$S/system/worker/etc.tar.gz" >/dev/null
}

sec_worker_files() {  # worker has no git checkout — its stack files are deployed copies
    rsync -a --exclude=.env "$WORKER:ai-stack/" "$S/worker/ai-stack/"
    rsync -a "$WORKER:Chatterbox-TTS-Server/"{config.yaml,voices,reference_audio} "$S/worker/chatterbox/"
    ssh "$WORKER" 'cd ~/Chatterbox-TTS-Server && git rev-parse HEAD && git diff' > "$S/local-mods/Chatterbox-TTS-Server.worker.diff"
}

sec_local_mods() {  # local patches to third-party checkouts on head (diffs, not copies)
    local r n
    for r in "$HOME/Chatterbox-TTS-Server" "$HOME/ComfyUI-Nvidia-Docker"; do
        n=$(basename "$r")
        { git -C "$r" rev-parse HEAD; git -C "$r" remote get-url origin; } > "$S/local-mods/$n.HEAD"
        git -C "$r" diff > "$S/local-mods/$n.diff"
        git -C "$r" status --porcelain > "$S/local-mods/$n.status"
    done
}

sec_comfyui() {
    rsync -a "$HOME/comfyui/output/" "$S/comfyui/output/"
    rsync -a "$HOME/comfyui/custom_nodes/" "$S/comfyui/custom_nodes/"
    rsync -a "$HOME/ComfyUI-Nvidia-Docker/run/ComfyUI/user/" "$S/comfyui/user/"
    (cd "$HOME/comfyui/models" && find . -type f -printf '%s\t%p\n' | sort -k2) > "$S/comfyui/models-manifest.tsv"
}


sec_repo() {  # git bundle covers unpushed commits / GitHub loss
    git -C "$STACK" bundle create "$S/repo/ai-stack.bundle" --all 2>/dev/null
    git -C "$STACK" bundle verify -q "$S/repo/ai-stack.bundle" >/dev/null 2>&1
    git -C "$STACK" status --porcelain > "$S/repo/uncommitted.txt"
    git -C "$STACK" log origin/master..master --oneline > "$S/repo/unpushed.txt" 2>/dev/null || true
    if [ -s "$S/repo/unpushed.txt" ]; then log "NOTE: $(wc -l < "$S/repo/unpushed.txt") commit(s) not pushed to GitHub"; fi
}

section "OpenWebUI (DB + Chroma via online backup, uploads)" sec_openwebui
section "Secrets head (.env, SSH, sparkrun config)"          sec_secrets_head
section "Secrets worker (SSH, .env)"                           sec_secrets_worker
section "System head (/etc, manifests)"                      sec_system_head
section "System worker (/etc, manifests)"                      sec_system_worker
section "worker deployed files + Chatterbox voices/config"     sec_worker_files
section "Local patches to third-party checkouts"               sec_local_mods
section "ComfyUI state + model manifest"                       sec_comfyui
section "Repo bundle"                                          sec_repo

# ---------------------------------------------------------------- write to each destination
save_image() {  # <host|""> <image> <dest dir>
    local host="$1" img="$2" dir="$3" id name
    local dk=(docker); [ -n "$host" ] && dk=(ssh "$host" docker)
    id=$("${dk[@]}" image inspect --format '{{.Id}}' "$img" 2>/dev/null) || { echo "image not present: $img" >&2; return 1; }
    name=$(echo "$img" | tr '/:' '__')
    if [ -f "$dir/$name.id" ] && [ "$(cat "$dir/$name.id")" = "$id" ] && [ -s "$dir/$name.tar.zst" ]; then
        log "  $img unchanged — skip"; return 0; fi
    log "  saving $img"
    if [ -n "$host" ]; then ssh "$host" "docker save '$img' | zstd -T0 -3 -q" > "$dir/.$name.tmp"
    else docker save "$img" | zstd -T0 -3 -q > "$dir/.$name.tmp"; fi
    zstd -tq "$dir/.$name.tmp" || { rm -f "$dir/.$name.tmp"; echo "corrupt archive: $img" >&2; return 1; }
    mv "$dir/.$name.tmp" "$dir/$name.tar.zst"; echo "$id" > "$dir/$name.id"
    # remote: the format string must reach the remote shell quoted, or ssh word-splits it
    if [ -n "$host" ]; then ssh "$host" "docker image inspect --format '{{join .RepoDigests \" \"}}' '$img'" > "$dir/$name.digest"
    else docker image inspect --format '{{join .RepoDigests " "}}' "$img" > "$dir/$name.digest"; fi
}

sync_comfy_models() {  # <dest>. No --delete: an emptied source must never wipe the backup copy.
    local D="$1" src_kb dst_kb
    src_kb=$(du -sk "$HOME/comfyui/models" | cut -f1)
    dst_kb=$(du -sk "$D/comfyui-models" 2>/dev/null | cut -f1 || echo 0)
    if [ "${dst_kb:-0}" -gt 0 ] && [ "$src_kb" -lt $((dst_kb / 2)) ]; then
        echo "source ~/comfyui/models is <50% of the backup copy — refusing to sync (restore it first?)" >&2; return 1
    fi
    rsync -a "$HOME/comfyui/models/" "$D/comfyui-models/"
}

IFS=: read -ra DESTS <<< "$BACKUP_DESTS"
WROTE=0
for D in "${DESTS[@]}"; do
    if [ ! -f "$D/$MARKER" ]; then
        warn "skipping $D — no $MARKER marker (drive/share not mounted? create the marker once on the real target)"; continue; fi
    log "→ $D"
    mkdir -p "$D/state"; chmod 700 "$D"
    PREV=$(ls -1d "$D"/state/20* 2>/dev/null | tail -1 || true)
    if ! rsync -a ${PREV:+--link-dest="$PREV"} "$S/" "$D/state/.$TS.partial/"; then warn "writing snapshot to $D failed"; continue; fi
    if [ -e "$D/state/$TS" ]; then warn "snapshot $TS already exists at $D"; continue; fi
    mv "$D/state/.$TS.partial" "$D/state/$TS"
    ls -1d "$D"/state/20* | head -n -"$KEEP" | xargs -r rm -rf
    WROTE=1
    if [ $IMAGES = 1 ]; then
        mkdir -p "$D/images"
        for i in "${SAVE_IMAGES_SPARK0[@]}"; do section "image (head) $i" save_image "" "$i" "$D/images"; done
        for i in "${SAVE_IMAGES_SPARK1[@]}"; do section "image (worker) $i" save_image "$WORKER" "$i" "$D/images"; done
    fi
    [ $COMFY_MODELS = 1 ] && section "ComfyUI models mirror" sync_comfy_models "$D"
    { echo "last run: $TS  status: $([ $FAIL = 0 ] && echo OK || echo WARNINGS — see journalctl -u ai-stack-backup)"
      du -sh "$D/state/$TS" "$D"/images "$D"/comfyui-models 2>/dev/null || true; } > "$D/LAST_RUN.txt"
done
[ $WROTE = 1 ] || warn "no snapshot written to ANY destination"

log "done ($([ $FAIL = 0 ] && echo OK || echo "with warnings — see above"))"
exit $FAIL
