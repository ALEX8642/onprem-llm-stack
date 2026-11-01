#!/usr/bin/env python3
"""Detect and repair the Qwen3.8 "!!!!" fault (systemd timer, every 15 min).

The fault (2026-10-11, on SGLang; the Atlas engine has not shown it, but the check is engine-neutral): the server keeps answering short prompts, but any prompt that crosses the
4,096-token prefill chunk with thinking OFF comes back as endless "!" (token 0 = NaN logits).
smoke-test style "Reply with: ok" checks stay green. flush_cache didn't clear it; a restart did.
Cause unknown (docs/STATUS.md, Known issues).

Each run:
  1. Skip quietly unless :8010 is serving Qwen3.8 and no model switch is in progress.
  2. Canary: ~6.5K tokens, thinking off, sent straight to SGLang. A random nonce goes FIRST so the
     prefix cache can't serve it (a cached canary is one small chunk and would pass while broken).
  3. On "!!!"/empty: save evidence, flush_cache, re-test. Still broken: restart via the model-control
     API (POST :9091/switch), at most once per 6 h. Broken again after that: leave it for a person.

Email (scripts/notify-failure.py settings in .env) at every step of an incident: fault detected,
then fixed-by-flush / restart triggered / needs a person, then recovered (with downtime) or "not back
after 25 min". Handled states exit 0; only a crash of this script exits non-zero (-> OnFailure email).
State + evidence: ~/ai-stack-incidents/  (NEEDS_ATTENTION = gave up; pending.json = restart in flight)
Manual run: python3 scripts/model-watchdog.py [--dry-run]   (--dry-run: detect + evidence, no repair)
Test:       WATCHDOG_TEST_FAULT=1 python3 scripts/model-watchdog.py --dry-run
"""
import json, os, secrets, subprocess, sys, time, urllib.request, fcntl
from datetime import datetime
from pathlib import Path

STACK = Path(__file__).resolve().parent.parent
SGLANG = "http://127.0.0.1:8010"
CONTROL = "http://127.0.0.1:9091"
MODEL_ID = "Qwen3.8-Flash-Next-NVFP4"
SWITCH_NAME = "qwen38atlas"  # restart target unless model-control reports the SGLang rollback running
QWEN38_TARGETS = {"qwen38atlas", "qwen38flash"}
LOCKFILE = "/tmp/ai-stack-model-switch.lock"
STATE = Path.home() / "ai-stack-incidents"
RESTART_COOLDOWN = 6 * 3600
WORKER = "worker"
DRY = "--dry-run" in sys.argv
TEST = bool(os.environ.get("WATCHDOG_TEST_FAULT"))
PENDING = STATE / "pending.json"
SLOW_RESTART = 25 * 60
HOW_TO = (f"Fix by hand (flush_cache exists on SGLang only): curl -X POST {SGLANG}/flush_cache ; if still broken:\n"
          f"  bash ~/ai-stack/scripts/stop-model.sh && bash ~/ai-stack/scripts/switch-model.sh {SWITCH_NAME}\n"
          f"Watchdog log: journalctl -u ai-stack-model-watchdog -n 50\n"
          f"Pause it: sudo systemctl disable --now ai-stack-model-watchdog.timer")


def log(msg):
    print(f"[watchdog] {msg}", flush=True)


def notify(subject, body):
    """Email via scripts/notify-failure.py (never raises; logs the outcome)."""
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location("notify", STACK / "scripts" / "notify-failure.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        log(mod.send_email(f"[ai-stack]{' [TEST]' if TEST else ''} {subject}",
                           f"{datetime.now():%Y-%m-%d %H:%M} on head\n\n{body}\n\n{HOW_TO}\n"))
    except Exception as e:
        log(f"notify failed: {e}")


def minutes(sec):
    return f"{sec/60:.0f} min"


def http(url, body=None, timeout=10, headers=None, post=False):
    data = json.dumps(body).encode() if body is not None else (b"" if post else None)
    req = urllib.request.Request(url, data, {"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode(errors="replace")


def switch_in_progress():
    fd = os.open(LOCKFILE, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def canary(thinking=False):
    """Returns (bad, prompt_tokens, content_preview). Raises on transport errors."""
    text = (STACK / "docs" / "MODELS.md").read_text()
    text = (text * 3)[:18000]
    prompt = f"[canary {secrets.token_hex(8)}]\n\n{text}\n\nIn one sentence, what is this document about?"
    body = {"model": MODEL_ID, "max_tokens": 24 if not thinking else 400, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": thinking}}
    d = http(f"{SGLANG}/v1/chat/completions", body, timeout=180)
    content = (d["choices"][0]["message"].get("content") or "")
    reasoning = (d["choices"][0]["message"].get("reasoning_content") or "")
    shown = content if (content or not thinking) else reasoning
    bad = "!!!" in shown or (not shown.strip())
    if os.environ.get("WATCHDOG_TEST_FAULT") and not thinking:  # test hook: pretend the fault is present
        bad = True
    return bad, d.get("usage", {}).get("prompt_tokens"), shown[:80]


def sh(cmd, timeout=30):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e:  # evidence collection must never stop the repair
        return f"<{e}>"


def save_evidence(incident, results):
    incident.mkdir(parents=True, exist_ok=True)
    (incident / "canary.json").write_text(json.dumps(results, indent=2))
    for name, cmd in {
        "uptime.txt": "uptime; docker ps --format '{{.Names}} {{.Status}}' | grep sparkrun",
        "metrics.txt": f"curl -s -m 10 {SGLANG}/metrics",
        "server_info.json": f"curl -s -m 10 {SGLANG}/get_server_info",
        "dmesg-head.txt": "sudo -n dmesg -T | tail -200",
        "dmesg-worker.txt": f"ssh -o BatchMode=yes {WORKER} 'sudo -n dmesg -T | tail -200'",
        "nvml.txt": "for c in $(docker ps --format '{{.Names}}' | grep sparkrun); do echo $c; docker exec $c nvidia-smi -L 2>&1; done; "
                    f"ssh -o BatchMode=yes {WORKER} \"for c in \\$(docker ps --format '{{{{.Names}}}}' | grep sparkrun); do echo \\$c; docker exec \\$c nvidia-smi -L 2>&1; done\"",
        "free.txt": f"free -m; ssh -o BatchMode=yes {WORKER} free -m",
    }.items():
        (incident / name).write_text(sh(cmd))
    sh(f"c=$(docker ps --format '{{{{.Names}}}}' | grep 'sparkrun.*node_0' | head -1); "
       f"docker cp $c:/tmp/sparkrun_serve.log {incident}/serve-node0.log", timeout=60)
    sh(f"ssh -o BatchMode=yes {WORKER} \"c=\\$(docker ps --format '{{{{.Names}}}}' | grep 'sparkrun.*node_1' | head -1); "
       f"docker exec \\$c cat /tmp/sparkrun_serve.log\" > {incident}/serve-node1.log", timeout=60)
    log(f"evidence saved: {incident}")


def main():
    STATE.mkdir(exist_ok=True)
    pending = json.loads(PENDING.read_text()) if PENDING.exists() else None
    try:
        served = [m["id"] for m in http(f"{SGLANG}/v1/models", timeout=5)["data"]]
    except Exception:
        served = None
    if served is None or MODEL_ID not in served or switch_in_progress():
        why = ("no server on :8010" if served is None else
               f":8010 serves {served}" if MODEL_ID not in served else "model switch in progress")
        if pending and not pending.get("slow_notified"):
            waited = time.time() - pending["restarted_at"]
            if waited > SLOW_RESTART:
                notify("Qwen3.8 NOT back after automatic restart",
                       f"The watchdog restarted the model {minutes(waited)} ago and it isn't serving yet ({why}).\n"
                       f"Startup log: /tmp/model-switch-current.log\nIncident: {pending['incident']}")
                pending["slow_notified"] = True
                PENDING.write_text(json.dumps(pending))
        log(f"{why}: skip")
        return 0

    try:
        bad, ptok, preview = canary()
    except Exception as e:
        log(f"canary request failed ({e}): skip, not treating a timeout as the fault")
        return 0
    if not bad:
        log(f"healthy ({ptok} prompt tokens, thinking off): {preview[:50]!r}")
        if pending:
            notify("Qwen3.8 RECOVERED after automatic restart",
                   f"Long prompts work again. Down about {minutes(time.time() - pending['since'])} "
                   f"(fault detected {pending['detected']}).\nIncident evidence: {pending['incident']}")
            PENDING.unlink()
        if (STATE / "NEEDS_ATTENTION").exists():
            (STATE / "NEEDS_ATTENTION").rename(STATE / f"resolved-{datetime.now():%Y%m%d-%H%M%S}")
            if not pending:
                notify("Qwen3.8 healthy again", "The watchdog's long-prompt check passes again; NEEDS_ATTENTION cleared.")
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    incident = STATE / stamp
    log(f"FAULT: {ptok} prompt tokens, thinking off -> {preview!r}")
    results = {"first": {"bad": bad, "prompt_tokens": ptok, "preview": preview}}
    try:  # settles "is thinking-on affected too?" next time it happens
        b2, p2, pv2 = canary(thinking=True)
        results["thinking_on"] = {"bad": b2, "prompt_tokens": p2, "preview": pv2}
        log(f"thinking-on canary: {'BAD' if b2 else 'ok'} {pv2[:50]!r}")
    except Exception as e:
        results["thinking_on"] = {"error": str(e)}
    save_evidence(incident, results)
    t_on = results["thinking_on"]
    seen = (f"Long-prompt check ({ptok} tokens, thinking off) returned: {preview!r}\n"
            f"Thinking-on check: {'BROKEN' if t_on.get('bad') else 'ok' if 'bad' in t_on else t_on.get('error')}\n"
            f"Evidence: {incident}")
    if pending:
        seen += f"\nThis is right after an automatic restart ({pending['detected']})."
    notify("Qwen3.8 FAULT detected: long prompts return \"!!!!\"",
           f"{seen}\n\n" + ("--dry-run: no repair attempted." if DRY else
                             "Repairing now: cache flush first, model restart if that doesn't work. Updates follow."))
    if DRY:
        log("--dry-run: no repair")
        return 0

    try:
        http(f"{SGLANG}/flush_cache", timeout=30, post=True)
    except Exception:
        pass
    try:
        bad, ptok, preview = canary()
    except Exception as e:
        bad, preview = True, f"<{e}>"
    results["after_flush"] = {"bad": bad, "preview": preview}
    (incident / "canary.json").write_text(json.dumps(results, indent=2))
    if not bad:
        log("fixed by flush_cache")
        notify("Qwen3.8 fixed by cache flush", f"The long-prompt check passes again after flush_cache.\n{seen}")
        if pending:
            PENDING.unlink()
        return 0

    last = STATE / "last_restart"
    since = time.time() - last.stat().st_mtime if last.exists() else None
    if since is not None and since < RESTART_COOLDOWN:
        (STATE / "NEEDS_ATTENTION").write_text(
            f"{stamp}: still broken after flush, and the last automatic restart was {since/3600:.1f} h ago.\n"
            f"Evidence: {incident}\n")
        log("still broken, restart cooldown active: NEEDS_ATTENTION written, leaving it for a person")
        notify("Qwen3.8 NEEDS ATTENTION: automatic repair gave up",
               f"Still broken after a cache flush, and the watchdog already restarted the model "
               f"{since/3600:.1f} h ago (limit: one automatic restart per {RESTART_COOLDOWN//3600} h).\n"
               f"It will keep checking and email when it's healthy again, but won't restart it.\n{seen}")
        if pending:
            PENDING.unlink()
        return 0

    token = ""
    for line in (STACK / ".env").read_text().splitlines():
        if line.startswith("MODEL_CONTROL_TOKEN="):
            token = line.split("=", 1)[1].strip().strip('"')
    auth = {"Authorization": f"Bearer {token}"}
    try:  # restart the build that is running (Atlas, or the SGLang rollback), not a fixed one
        target = http(f"{CONTROL}/status", timeout=20, headers=auth).get("active_model")
    except Exception:
        target = None
    target = target if target in QWEN38_TARGETS else SWITCH_NAME
    try:
        r = http(f"{CONTROL}/switch", {"model": target}, timeout=15, headers=auth)
    except Exception as e:
        r = {"error": str(e)}
    results["restart"] = r
    (incident / "canary.json").write_text(json.dumps(results, indent=2))
    if isinstance(r, dict) and r.get("status") == "started":
        last.write_text(stamp)
        last.touch()
        PENDING.write_text(json.dumps({"incident": str(incident), "detected": stamp,
                                       "since": time.time(), "restarted_at": time.time()}))
        log(f"still broken after flush: model restart ({target}) triggered via model-control /switch")
        notify("Qwen3.8 restarting (automatic)",
               f"Still broken after a cache flush, so the watchdog restarted the model (takes ~3 min on Atlas, ~8 on SGLang; "
               f"chat is unavailable meanwhile). You'll get a RECOVERED email, or a warning if it isn't "
               f"back within {minutes(SLOW_RESTART)}.\n{seen}")
        return 0
    (STATE / "NEEDS_ATTENTION").write_text(f"{stamp}: restart request failed: {r}\nEvidence: {incident}\n")
    log(f"restart request failed: {r}")
    notify("Qwen3.8 NEEDS ATTENTION: restart request failed",
           f"Still broken after a cache flush, and the restart request to model-control failed: {r}\n{seen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
