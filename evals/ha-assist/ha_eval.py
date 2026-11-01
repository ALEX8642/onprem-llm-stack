#!/usr/bin/env python3
"""Offline evaluation of the Home Assistant voice agent's tool calls. Nothing is executed.

Replays labelled utterances with the exact request Home Assistant's extended_openai_conversation
sends (system prompt rendered from a frozen snapshot of the exposed entities, the execute_services
tool, HA's sampling), then compares the returned tool_calls with the labels. Stdlib only.

  run    ha_eval.py run --base URL --model NAME [--key K] [--repeats 3] --out results/x.json
  score  ha_eval.py score results/x.json [...]        re-score saved raw responses against labels.jsonl

Scoring rules (see README.md):
  action items   pass when every expected slot is met and no call falls outside the slots
                 (accept_none items also pass with no call: the target state already holds)
  none/clarify/refuse items   pass when no tool call is made
  malformed_call   a tool call whose arguments don't parse into service calls (fails either way)
  context_gap items are scored, and also reported separately: what HA sends can't answer them
"""
import argparse, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixture")
NOW = "2026-10-29 19:30:00.000000-04:00"  # frozen so every run sees the same prompt


def load_fixture():
    ents = json.load(open(os.path.join(FIX, "entities.json")))
    tmpl = open(os.path.join(FIX, "prompt.j2")).read()
    funcs = json.load(open(os.path.join(FIX, "functions.json")))
    req = json.load(open(os.path.join(FIX, "request.json")))
    return ents, tmpl, funcs, req


def render_prompt(tmpl, ents):
    """Renders the two Jinja expressions HA's prompt uses, matching its whitespace control."""
    start = tmpl.index("{% for entity in exposed_entities -%}")
    end_tag = "{% endfor -%}"
    end = tmpl.index(end_tag)
    rows = "".join(f"{e['entity_id']},{e['name']},{e['state']},{'/'.join(e['aliases'])}\n" for e in ents)
    after = tmpl[end + len(end_tag):].lstrip()
    out = tmpl[:start] + rows + after
    assert "{%" not in out and "{{ entity" not in out
    return out.replace("{{now()}}", NOW)


def load_labels():
    return [json.loads(l) for l in open(os.path.join(HERE, "labels.jsonl")) if l.strip()]


def post(base, key, body, timeout=300):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.time() - t0


def cmd_run(a):
    ents, tmpl, funcs, req = load_fixture()
    system = render_prompt(tmpl, ents)
    labels = load_labels()
    if a.only:
        labels = [l for l in labels if l["id"] in a.only.split(",")]
    jobs = [(l, rep) for rep in range(a.repeats) for l in labels]

    def one(job):
        l, rep = job
        body = {"model": a.model, "messages": [{"role": "system", "content": system},
                                               {"role": "user", "content": l["utterance"]}],
                "top_p": req["top_p"], "temperature": req["temperature"], "max_tokens": req["max_tokens"],
                "user": f"eval-{l['id']}-{rep}",
                "tools": [{"type": "function", "function": f} for f in funcs], "tool_choice": req["tool_choice"]}
        try:
            d, dt = post(a.base, a.key, body)
            m = d["choices"][0]["message"]
            return {"id": l["id"], "rep": rep, "latency": round(dt, 3), "finish": d["choices"][0].get("finish_reason"),
                    "content": m.get("content") or "", "reasoning_chars": len(m.get("reasoning_content") or m.get("reasoning") or ""),
                    "tool_calls": m.get("tool_calls") or [], "usage": d.get("usage")}
        except Exception as e:  # an error is a result, not a crash
            return {"id": l["id"], "rep": rep, "error": f"{type(e).__name__}: {e}"}

    raw = []
    with ThreadPoolExecutor(a.concurrency) as ex:
        for i, r in enumerate(ex.map(one, jobs), 1):
            raw.append(r)
            print(f"[{i}/{len(jobs)}] {r['id']} rep{r['rep']} {r.get('latency', r.get('error'))}", flush=True)
    out = {"model": a.model, "base": a.base, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "repeats": a.repeats, "concurrency": a.concurrency, "system_prompt_chars": len(system),
           "request": {k: req[k] for k in ("temperature", "top_p", "max_tokens", "tool_choice")}, "raw": raw}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    report(a.out)


# ---------- scoring

def calls_of(r):
    """Flattens execute_services tool calls into a set of (service, entity_id) pairs."""
    pairs, bad = set(), False
    for tc in r.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            bad = True
            continue
        items = args.get("list") if isinstance(args, dict) else None
        if not isinstance(items, list):
            bad = True
            continue
        for it in items:
            dom, svc = it.get("domain", ""), it.get("service", "")
            ent = (it.get("service_data") or {}).get("entity_id", "")
            ents = ent if isinstance(ent, list) else [e.strip() for e in str(ent).split(",") if e.strip()]
            for e in ents or [""]:
                d = e.split(".")[0] if dom == "homeassistant" and "." in e else dom  # homeassistant.turn_off == <domain>.turn_off
                pairs.add((f"{d}.{svc}", e))
    return pairs, bad


def judge(label, r, all_lights):
    if "error" in r:
        return "error"
    pairs, bad = calls_of(r)
    exp = label["expect"]
    if exp["type"] != "action":
        return "unnecessary_action" if pairs else "malformed_call" if bad else "pass"
    if not pairs:
        if bad or "execute_services" in r.get("content", ""):
            return "malformed_call"
        return "pass" if label.get("accept_none") else "missed"
    allowed = set()
    for s in exp["slots"]:
        opt = all_lights + ["all"] if s["optional"] == "ALL_LIGHTS" else s["optional"]
        for ents in s["one_of"]:
            allowed |= {(s["service"], e) for e in ents}
        allowed |= {(s["service"], e) for e in opt}
    met = all(any({(s["service"], e) for e in ents} <= pairs for ents in s["one_of"]) for s in exp["slots"])
    if met and pairs <= allowed:
        return "pass"
    services_ok = {p[0] for p in pairs} <= {s["service"] for s in exp["slots"]}
    return "wrong_entity" if services_ok else "wrong_service"


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def score(res, labels, ents):
    all_lights = [e["entity_id"] for e in ents if e["entity_id"].startswith("light.")]
    L = {l["id"]: l for l in labels}
    rows = [dict(r, verdict=judge(L[r["id"]], r, all_lights), cat=L[r["id"]]["cat"], type=L[r["id"]]["expect"]["type"])
            for r in res["raw"] if r["id"] in L]
    def rate(sel, ok):
        sel = list(sel)
        return {"n": len(sel), "rate": round(sum(1 for r in sel if ok(r)) / len(sel), 3) if sel else None}
    action = [r for r in rows if r["type"] == "action"]
    noact = [r for r in rows if r["type"] != "action"]
    lat = [r["latency"] for r in rows if "latency" in r]
    s = {
        "overall_accuracy": rate(rows, lambda r: r["verdict"] == "pass"),
        "tool_call_exact_match": rate(action, lambda r: r["verdict"] == "pass"),
        "wrong_entity_rate": rate(action, lambda r: r["verdict"] == "wrong_entity"),
        "missed_action_rate": rate(action, lambda r: r["verdict"] in ("missed", "malformed_call")),
        "malformed_call_rate": rate(rows, lambda r: r["verdict"] == "malformed_call"),
        "unnecessary_action_rate": rate(noact, lambda r: r["verdict"] == "unnecessary_action"),
        "refusal_correctness": rate([r for r in rows if r["type"] == "refuse"], lambda r: r["verdict"] == "pass"),
        "clarify_correctness": rate([r for r in rows if r["type"] == "clarify"], lambda r: r["verdict"] == "pass"),
        "errors": sum(1 for r in rows if r["verdict"] == "error"),
        "latency_s": {"p50": pct(lat, 0.5), "p95": pct(lat, 0.95)},
        "by_category": {c: rate([r for r in rows if r["cat"] == c], lambda r: r["verdict"] == "pass")
                        for c in sorted({r["cat"] for r in rows})},
    }
    gap = [r for r in rows if not L[r["id"]].get("context_gap")]
    s["overall_accuracy_excl_context_gap"] = rate(gap, lambda r: r["verdict"] == "pass")
    s["tool_call_exact_match_excl_context_gap"] = rate([r for r in gap if r["type"] == "action"], lambda r: r["verdict"] == "pass")
    # an item is stable when every repeat got the same verdict
    by_id = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r["verdict"])
    s["unstable_items"] = sorted(i for i, v in by_id.items() if len(set(v)) > 1)
    misses = [{"id": r["id"], "rep": r["rep"], "utterance": L[r["id"]]["utterance"], "verdict": r["verdict"],
               "context_gap": bool(L[r["id"]].get("context_gap")), "calls": sorted(map(list, calls_of(r)[0])) if "error" not in r else r["error"],
               "content": (r.get("content") or "")[:160]} for r in rows if r["verdict"] != "pass"]
    return {"summary": s, "misses": misses}


def report(path):
    res = json.load(open(path))
    ents, *_ = load_fixture()
    sc = score(res, load_labels(), ents)
    res["score"] = sc
    json.dump(res, open(path, "w"), indent=1)
    print(f"\n== {res['model']}  ({path})")
    print(json.dumps(sc["summary"], indent=1))
    print(f"{len(sc['misses'])} misses (all listed in the results file)")


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    sp = P.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--base", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--key", default=os.environ.get("EVAL_KEY", "none"))
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--concurrency", type=int, default=1)
    r.add_argument("--only", default="")
    r.add_argument("--out", required=True)
    s = sp.add_parser("score")
    s.add_argument("paths", nargs="+")
    a = P.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        for p in a.paths:
            report(p)
