#!/usr/bin/env python3
"""Backend-agnostic A/B benchmark for an OpenAI-compatible chat endpoint.

Built 2026-09-26 to compare Qwen3.8-Flash-Next recipes (SGLang vs vLLM) on the same prompts,
so a faster recipe can't hide a regression. Stdlib only. Measures:
  speed      single-stream decode tok/s + TTFT, thinking off (8 categories) and on (2 prompts)
  concurrent aggregate + per-stream tok/s at x2 and x4
  multiturn  TTFT of a follow-up turn on a ~24K-token prefix (does prefix caching work?)
  needle     ~100K-token prompt, needle retrieval + prefill rate
  tools      tool call parsed into tool_calls (not leaked into content)
  reasoning  thinking-on puts reasoning in reasoning_content, not content
  quiz       12 auto-graded questions, thinking on (quality check)
  vision     one tiny image (reports whether the endpoint accepts images at all)

Usage: bench-llm.py --base http://head:8010/v1 --model NAME --out results.json [--only speed,quiz]
"""
import argparse, base64, json, os, re, struct, sys, threading, time, urllib.request, zlib

P = argparse.ArgumentParser()
P.add_argument("--base", required=True)
P.add_argument("--model", required=True)
P.add_argument("--key", default=os.environ.get("LITELLM_KEY", "none"))
P.add_argument("--out", required=True)
P.add_argument("--only", default="")
A = P.parse_args()


def post(body, stream=False, timeout=900):
    req = urllib.request.Request(A.base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {A.key}"})
    return urllib.request.urlopen(req, timeout=timeout)


def chat(messages, think, max_tokens=1024, stream=True, **extra):
    """Streams a completion. Returns dict with ttft, decode tok/s, content, reasoning, tool_calls, usage."""
    body = {"model": A.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.6,
            "top_p": 0.95, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": think}, **extra}
    if stream:
        body["stream_options"] = {"include_usage": True}
    t0 = time.time()
    first = None
    content, reasoning, tools, usage = [], [], {}, None
    with post(body, stream) as r:
        if not stream:
            d = json.load(r)
            m = d["choices"][0]["message"]
            return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
                    "tool_calls": m.get("tool_calls") or [], "usage": d.get("usage"), "total": time.time() - t0}
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content") or ""
                rpiece = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if (piece or rpiece or delta.get("tool_calls")) and first is None:
                    first = time.time()
                content.append(piece)
                reasoning.append(rpiece)
                for tc in delta.get("tool_calls") or []:
                    t = tools.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    t["name"] += fn.get("name") or ""
                    t["arguments"] += fn.get("arguments") or ""
    end = time.time()
    ntok = (usage or {}).get("completion_tokens") or 0
    first = first or end
    return {"ttft": first - t0, "decode_tps": ntok / max(end - first, 1e-6) if ntok > 1 else 0.0,
            "completion_tokens": ntok, "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "content": "".join(content), "reasoning": "".join(reasoning),
            "tool_calls": list(tools.values()), "total": end - t0}


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


SPEED_PROMPTS = {
    "prose": "Write a warm, 250-word letter to a grandparent describing a family camping trip.",
    "narrative": "Tell a 300-word short story about a lighthouse keeper who finds a message in a bottle.",
    "summary": "Summarize the causes and consequences of the French Revolution in about 200 words.",
    "coding": "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H5M' into seconds, with docstring and 5 unit tests.",
    "math": "A train leaves at 3:15pm going 72 km/h; a second leaves the same station at 4:00pm going 90 km/h on the same track. When and where does the second catch the first? Show your work.",
    "json": "Return a JSON array of 8 fictional employees, each with id, name, department, start_date (ISO), salary, and a skills array. JSON only.",
    "html": "Write a complete single-file HTML page with CSS for a responsive pricing table with three tiers.",
    "format": "Produce a markdown table comparing 6 programming languages on typing, speed, memory safety, and primary use.",
}

QUIZ = [  # (question, regex the final answer must match)
    ("What is 17 * 23? Answer with just the number.", r"\b391\b"),
    ("How many r's are in the word 'strawberry'? Answer with just the number.", r"\b3\b"),
    ("What is the capital of Australia? One word.", r"canberra"),
    ("If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Answer yes or no.", r"\byes\b"),
    ("What is the derivative of x^3 * sin(x)? Give the expression.", r"3x\^?2.*sin.*x\^?3.*cos|x\^?3.*cos.*3x\^?2.*sin"),
    ("A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How many cents does the ball cost? Just the number.", r"\b5\b"),
    ("What is the smallest prime number greater than 100? Just the number.", r"\b101\b"),
    ("Which planet has the most confirmed moons as of 2024: Jupiter or Saturn? One word.", r"saturn"),
    ("Sort these numbers ascending and output them comma-separated: 42, 7, 19, 3, 88, 21", r"3,\s*7,\s*19,\s*21,\s*42,\s*88"),
    ("What is the sum of the interior angles of a hexagon, in degrees? Just the number.", r"\b720\b"),
    ("Convert binary 101101 to decimal. Just the number.", r"\b45\b"),
    ("Who wrote 'Pride and Prejudice'? Full name.", r"jane austen"),
]

FILLER = ("The committee reviewed the quarterly logistics report in detail, noting shipment volumes, "
          "warehouse throughput, staffing levels, and seasonal demand patterns across all regions. ")


def filler_tokens(n_tokens):
    return FILLER * max(1, n_tokens // 30)  # ~30 tokens per sentence


def tiny_png():
    # 8x8 solid red PNG
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * 8 for _ in range(8))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


R = {"base": A.base, "model": A.model, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
only = set(filter(None, A.only.split(",")))


def want(name):
    return not only or name in only


def log(*a):
    print(*a, flush=True)


def section(name, fn):
    if not want(name):
        return
    log(f"== {name}")
    try:
        R[name] = fn()
    except Exception as e:  # keep going; a failed section is itself a finding
        R[name] = {"error": f"{type(e).__name__}: {e}"}
    log(json.dumps(R[name], indent=1)[:1500])
    with open(A.out, "w") as f:
        json.dump(R, f, indent=1)


def s_speed():
    chat([{"role": "user", "content": "Say hi."}], False, 16)  # warmup
    out = {"off": {}, "on": {}}
    for k, p in SPEED_PROMPTS.items():
        r = chat([{"role": "user", "content": p}], False, 1024)
        out["off"][k] = {"tps": round(r["decode_tps"], 1), "ttft": round(r["ttft"], 3), "tok": r["completion_tokens"]}
    for k in ("math", "coding"):
        r = chat([{"role": "user", "content": SPEED_PROMPTS[k]}], True, 4096)
        out["on"][k] = {"tps": round(r["decode_tps"], 1), "ttft": round(r["ttft"], 3), "tok": r["completion_tokens"]}
    out["off_median_tps"] = med([v["tps"] for v in out["off"].values()])
    out["on_median_tps"] = med([v["tps"] for v in out["on"].values()])
    return out


def s_concurrent():
    out = {}
    keys = list(SPEED_PROMPTS)
    for n in (2, 4):
        res = [None] * n
        def run(i):
            res[i] = chat([{"role": "user", "content": SPEED_PROMPTS[keys[i % len(keys)]]}], False, 600)
        t0 = time.time()
        th = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        [t.start() for t in th]; [t.join() for t in th]
        wall = time.time() - t0
        out[f"x{n}"] = {"per_stream_median_tps": round(med([r["decode_tps"] for r in res]), 1),
                        "aggregate_tps": round(sum(r["completion_tokens"] for r in res) / wall, 1),
                        "max_ttft": round(max(r["ttft"] for r in res), 3)}
    return out


def s_multiturn():
    doc = filler_tokens(24000) + "\nThe project codename mentioned in the appendix is BLUEHERON.\n"
    m = [{"role": "user", "content": doc + "\nWhat is this document about, in one sentence?"}]
    r1 = chat(m, False, 100)
    m += [{"role": "assistant", "content": r1["content"]},
          {"role": "user", "content": "What is the project codename? One word."}]
    r2 = chat(m, False, 30)
    return {"prompt_tokens": r1["prompt_tokens"], "turn1_ttft": round(r1["ttft"], 2),
            "turn2_ttft": round(r2["ttft"], 2), "turn2_correct": "BLUEHERON" in r2["content"].upper(),
            "prefix_reuse_speedup": round(r1["ttft"] / max(r2["ttft"], 1e-3), 1)}


def s_needle():
    half = filler_tokens(50000)
    doc = half + "\nNOTE: the vault combination is 7-3-9-1-4.\n" + half
    r = chat([{"role": "user", "content": doc + "\nWhat is the vault combination? Answer with just the digits."}], False, 30)
    return {"prompt_tokens": r["prompt_tokens"], "ttft": round(r["ttft"], 1),
            "prefill_tps": round((r["prompt_tokens"] or 0) / max(r["ttft"], 1e-3)),
            "correct": bool(re.search(r"7\D*3\D*9\D*1\D*4", r["content"])), "answer": r["content"][:80]}


TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                       "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}


def s_tools():
    out = {}
    for think in (False, True):
        r = chat([{"role": "user", "content": "What's the weather in Toronto in celsius?"}], think, 2048,
                 tools=[TOOL], tool_choice="auto")
        tc = r["tool_calls"][0] if r["tool_calls"] else None
        args = {}
        try:
            args = json.loads(tc["arguments"]) if tc else {}
        except Exception:
            pass
        out["think_on" if think else "think_off"] = {
            "parsed": bool(tc) and tc["name"] == "get_weather" and "toronto" in str(args.get("city", "")).lower(),
            "leaked_into_content": "get_weather" in r["content"] or "<tool_call" in r["content"],
            "raw": (tc or {}), "content": r["content"][:200]}
    return out


def s_reasoning():
    r = chat([{"role": "user", "content": "Is 221 prime? Answer yes or no."}], True, 2048)
    return {"reasoning_chars": len(r["reasoning"]), "content": r["content"][:200],
            "think_tags_in_content": "<think>" in r["content"] or "</think>" in r["content"]}


def s_quiz():
    out, score = [], 0
    for q, rx in QUIZ:
        r = chat([{"role": "user", "content": q}], True, 4096)
        ok = bool(re.search(rx, r["content"].lower().replace("**", "").replace("$", "").replace(" ", " ")))
        score += ok
        out.append({"q": q[:50], "ok": ok, "a": r["content"].strip()[-120:]})
    return {"score": f"{score}/{len(QUIZ)}", "items": out}


def s_vision():
    url = "data:image/png;base64," + base64.b64encode(tiny_png()).decode()
    r = chat([{"role": "user", "content": [{"type": "text", "text": "What single color fills this image? One word."},
                                           {"type": "image_url", "image_url": {"url": url}}]}], False, 20)
    return {"answer": r["content"][:80], "correct": "red" in r["content"].lower()}


for name, fn in [("speed", s_speed), ("concurrent", s_concurrent), ("multiturn", s_multiturn),
                 ("needle", s_needle), ("tools", s_tools), ("reasoning", s_reasoning),
                 ("quiz", s_quiz), ("vision", s_vision)]:
    section(name, fn)
R["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
with open(A.out, "w") as f:
    json.dump(R, f, indent=1)
log("done ->", A.out)
