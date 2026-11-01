#!/usr/bin/env bash
# Sample both nodes' memory every N seconds into a CSV, for bench runs (bench-results/*-mem.csv).
# Columns match the 2026-10-24 Atlas/SGLang A/B: ts,node,used_mib,avail_mib,swap_used_mib,gpu_proc_mib
# (gpu_proc_mib = summed per-process GPU memory from nvidia-smi; GB10 has no device-level memory query).
# Usage: mem-sample.sh OUT.csv [interval_s]   — runs until killed.
set -euo pipefail
OUT="$1"; INTERVAL="${2:-5}"
NODES=(head worker)
sample='free -m | awk "/^Mem:/{u=\$3; a=\$7} /^Swap:/{s=\$3} END{printf \"%s,%s,%s\", u, a, s}"; echo -n ,; nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk "{t+=\$1} END{print t+0}"'
[ -s "$OUT" ] || echo "ts,node,used_mib,avail_mib,swap_used_mib,gpu_proc_mib" > "$OUT"
while true; do
    ts=$(date +%s)
    for n in "${NODES[@]}"; do
        if [ "$n" = "head" ]; then line=$(bash -c "$sample"); else line=$(ssh -o BatchMode=yes "$n" "$sample"); fi
        echo "$ts,$n,$line" >> "$OUT"
    done
    sleep "$INTERVAL"
done
