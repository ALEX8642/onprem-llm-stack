# Command A+ W4A4 on two DGX Spark (GB10) nodes

Status: **done 2026-10-29. Verdict: runs well on two GB10s, but not a replacement here.**
Experiment only: production went straight back to Qwen3.8-Flash-Next on Atlas.

Model: [`CohereLabs/command-a-plus-05-2026-w4a4`](https://huggingface.co/CohereLabs/command-a-plus-05-2026-w4a4)
at revision `ebd2d72`. Apache 2.0, 218B total / 25B active, 128 experts (8 routed + shared), NVFP4
W4A4 on the MoE experts only; attention, KV cache and the vision tower stay BF16. The card's minimum
is one B200 at TP=1. Here it runs TP=2 across two GB10 nodes (128 GB unified memory each, about
121.7 GB usable).

## 1. Feasibility (checked before downloading anything)

| Question | Answer |
|---|---|
| vLLM >= 0.25 for arm64 + GB10 on driver 580.159.03? | **Yes**, already on both nodes: `eugr/spark-vllm` (vLLM 0.26.1rc1, Aug 2026), the same image family that serves MiniMax M2.7 NVFP4 here. A second community image has 0.25.2. No driver, kernel or CUDA change. |
| Does that vLLM know the model? | **Yes**: `Cohere2MoeForCausalLM`, `Cohere2VisionForConditionalGeneration`, and the `cohere_command4` tool and reasoning parsers are all registered. |
| Does `cohere_melody` (>= 0.9.0) install on arm64? | **Yes**: 0.15.3 has a cp312 aarch64 wheel and no dependencies, so adding it changes nothing else in the image. |
| Can sparkrun run a custom image at TP=2? | **Yes**: `--image` on a recipe file, the same path an earlier DeepSeek trial used. Ray is already in the base image. |

Image: the base pinned by digest plus one `pip install cohere_melody==0.15.3` layer
(`scripts/recipes/Dockerfile.command-a-plus`).

### Benchmark fairness fixes found during setup

The A/B harness (`scripts/bench-llm.py`) was written for Qwen and would have measured Command A+
wrongly in two ways:
- **Thinking switch.** It turns thinking off with `chat_template_kwargs.enable_thinking=false`.
  Command A+'s template ignores that key; it reads `reasoning` (or `reasoning_effort: "none"`), so
  every "thinking off" run would silently have been thinking. Added `--think-kwarg reasoning`.
- **Sampling.** It forces temperature 0.6 / top_p 0.95 on every model. That is Qwen's
  recommendation, not Cohere's (0.9 / 0.95, repetition penalty 1.04, shipped in the checkpoint's
  `generation_config.json`). Added `--sampling server`, which sends none and lets vLLM apply the
  model's own. Each model runs with its own recommended settings.

Defaults are unchanged, so the Qwen baseline runs exactly as before. The leak checks also look for
Command A+'s `<|START_THINKING|>` / `<|START_ACTION|>` markers.

## 2. Memory budget (estimate, to be checked against the measured numbers)

Weights: 122.8 GB by sparkrun's count, **about 61.4 GB per node** at TP=2.

KV cache (BF16; 8 KV heads × 128 head_dim = 4 KiB per token per layer; TP=2 halves it per node).
32 layers, but only 8 are global attention; the other 24 use a 4,096-token sliding window:

| Context | Global layers only (+ fixed 4K window) | All 32 layers full (worst case, sparkrun's estimate) |
|---|---|---|
| 32K | 0.5 + 0.2 GB per node per sequence | 2 GB |
| 96K | 1.5 + 0.2 GB | 6 GB |
| 128K | 2.0 + 0.2 GB | 8 GB |

Reference point: MiniMax M2.7 NVFP4 (130 GB of weights, 62 layers) needs gpu-mem **0.70** (about
84.7 GB per node) for 96K × 4 sequences, about 4 GB of it vLLM's CUDA-graph profiling. Command A+
has slightly smaller weights and much cheaper KV, so the expectation is a floor of about **0.62–0.66**
for 128K × 4, and it starts at 0.70. That is no more than MiniMax, under which FLUX ran alongside, so unlike
the original plan's assumption, ComfyUI may well fit next to it. To be measured.

Speed expectation, so the result isn't a surprise: decode on GB10 is memory-bandwidth-bound (about
273 GB/s per node). Per token, each node reads its half of the BF16 attention weights (about 4.5 GB)
plus its half of the active NVFP4 expert weights (about 5–6 GB), so the ceiling is roughly
25 tok/s single-stream, and 12–20 is realistic. The current Qwen3.8 (6B active) decodes at 67.

## 3. Setup and what broke

Everything ran on the pinned driver (580.159.03); no driver, kernel, CUDA or `apt` change.

- **Launch: first try.** vLLM picked FlashInfer CUTLASS for the NVFP4 MoE experts and the NVFP4
  linear layers, and FlashAttention for attention. First cold start 972 s (it copied the image and
  132 GB of weights to the second node); container to ready 520–545 s after that.
- **"Thinking off" isn't honoured.** With `reasoning: false` the template correctly ends the prompt
  with an empty `<|START_THINKING|><|END_THINKING|>` block, but the model reasons anyway, then emits
  a second `<|END_THINKING|>` and the answer (checked on raw tokens via `/v1/completions`). Because
  the prompt says thinking is closed, vLLM's `cohere_command4` reasoning parser then passes that
  reasoning through as the *answer*. So every "thinking off" row below is really thinking-on
  decoding, and some thinking-off quality checks fail for that reason alone (rerun with thinking on,
  below). Streaming also leaked `<|START_TEXT|>`/`<|END_TEXT|>` into `content` once.
- **Degenerate output on a synthetic repetitive prompt.** The harness's multi-turn test is one
  sentence repeated ~800 times (about 22K tokens) with a codename at the end. Command A+ failed it
  12 times out of 14 across every sampling setting tried (with and without the repetition penalty,
  temperature 0.6 and 0.9, cold and prefix-cached), sometimes with garbage output
  (`clean_border = d< /d> /d>`, echoing the filler) even when its reasoning had found the answer.
  On ~21K tokens of **real** documentation it answered 6/6 recall questions correctly, cold, and the
  93K needle passed with thinking on. So read the multi-turn row as "breaks on pathological
  repetition", not "can't do 22K".
- **Tooling gotcha (ours, not the model's).** `switch-model.sh` treats any non-empty cache
  directory as "already cached". A 5.6 MB config-only stub left by `sparkrun recipe vram` made it skip
  the 132 GB modelstore copy, and sparkrun would have re-downloaded the weights from the internet.
  Caught and fixed by hand; proposed fix: check for `*.safetensors` (or the snapshot's size) instead
  of "non-empty".
- Each node built its own copy of the image, so sparkrun saw different image IDs and re-synced one
  node's image. Harmless; building once and letting sparkrun distribute it avoids it.

## 4. Measured memory (per node, sampled every 5 s; `bench-results/2026-10-29-*-mem.csv`)

vLLM's own accounting: weights **61.48 GiB** per node (the estimate was 61.4 GB).

| gpu-mem | KV cache | KV tokens | Full-length (128K) requests at once |
|---|---|---|---|
| 0.70 | 22.4 GiB | 1,128,719 | 8.6 |
| 0.60 | 9.97 GiB | 482,518 | 3.7 |

The hybrid KV cache works: 24 of the 32 layers only hold a 4K sliding window, so the KV cost per
token is about a fifth of a naive 32-layer estimate. **The practical floor is about 0.60**, and
0.62 guarantees 4 × 128K.

| Node RAM | Qwen3.8 Atlas 0.60 (production) | Command A+ 0.70 | Command A+ 0.70 + FLUX | Command A+ 0.60 (+ FLUX) |
|---|---|---|---|---|
| peak used, node 1 / node 2 | 80.2 / 78.9 GiB | 99.5 / 101.8 | **114.0** / 101.8 | 104.7 / 101.8 |
| lowest available, node 1 | 41.4 GiB | 22.2 | **7.6** | 16.9 |
| peak swap, node 1 / node 2 | 4.4 / 3.1 GiB | 8.9 / 5.0 | 7.9 / 5.0 | 8.4 / 6.1 |

**ComfyUI does fit alongside it.** FLUX.1-schnell generated a 1024² image in 21.0 s at 0.70 and in
18.8 s at 0.60 (17–18.5 s next to Qwen), with no OOM, and chat kept answering with FLUX resident. At
0.70 the margin was thin (7.6 GiB available at the worst moment); 0.60 left 16.9 GiB.

## 5. Benchmark vs Qwen3.8 (same harness, same day)

Harness: `scripts/bench-llm.py` against each server directly. Each model with its own recommended
sampling (Qwen 0.6 / 0.95; Command A+ its generation_config 0.9 / 0.95 / repetition penalty 1.04)
and its own thinking switch. Qwen3.8 runs on the Atlas engine with MTP speculative decoding;
Command A+ on vLLM without speculation. Results: `bench-results/2026-10-29-{qwen38-atlas,command-a-0.70}.json`.

| | Qwen3.8-Flash-Next NVFP4 (Atlas) | Command A+ W4A4 (vLLM) |
|---|---|---|
| active / total params | 6B / 125B (+ n-gram table) | 25B / 218B |
| decode median, "thinking off" / on | **73.0 / 76.4 tok/s** | 18.5 / 18.3 tok/s |
| prose / coding | 44.1 / 83.6 | 18.5 / 18.4 (flat: no speculation) |
| x2 / x4 aggregate | **66.5 / 116.2** | 28.7 / 38.1 |
| x4 per stream | 31.2 | 10.7 |
| 93K needle: prefill / correct | **3,482 tok/s** / yes | 2,320 tok/s / no with thinking "off" (answer leak), **yes with thinking on** |
| 22K follow-up TTFT (prefix cache) | 0.31 s | 0.36 s |
| multi-turn recall (repetitive filler) | correct | wrong (see section 3) |
| tool call parsed into `tool_calls` | 2/2, no leak | 2/2; "thinking off" leaked reasoning into `content` |
| reasoning split | clean | clean in content, but `<|START_TEXT|>` tags leaked when streaming |
| quiz (12, thinking on) | 12/12 | 11/12; the miss is a correct derivative in LaTeX (`x^{2}`) the grader's regex doesn't accept, so effectively 12/12 |
| vision (8×8 red PNG) | correct | wrong with thinking "off" (reasoning leaked), **correct with thinking on** |

Decode speed is close to the estimate: memory-bandwidth-bound at about 18.5 tok/s, against the ~25
tok/s ceiling for reading 25B active params (BF16 attention + NVFP4 experts) per token.

## 6. Tool calling

- **Parser works.** With `--tool-call-parser cohere_command4 --enable-auto-tool-choice`, tool calls
  arrive as proper OpenAI `tool_calls` with valid JSON arguments (bench `tools`, 2/2; the Home
  Assistant replay below, 183 requests).
- **Agentic accuracy on a real tool, though, was below Qwen's.** Replaying the actual Home Assistant
  voice request (one `execute_services` tool, 176 exposed entities, 61 labelled utterances × 3,
  nothing executed; `evals/ha-assist/`):

| | Qwen3.8 (Atlas) | Command A+ |
|---|---|---|
| overall accuracy | **83.1%** | 69.9% |
| tool-call exact match | **91.2%** | 74.5% |
| wrong-entity rate | **5.9%** | 10.8% |
| unnecessary-action rate | **23.5%** | 30.9% |
| refusal correctness | **57%** | 33% |
| asks when ambiguous | **40%** | 20% |
| malformed calls | **1.6%** | 3.3% |
| latency p50 / p95 | **3.1 / 15.5 s** | 13.8 / 37.6 s |

  Re-scored 2026-11-01 against corrected labels ("Red alert!" is meant to fire its scene); the model
  outputs are unchanged. See `evals/ha-assist/README.md` for a fresh Qwen3.8 run after the risky
  switches were unexposed.

  Command A+'s characteristic misses: invented services (`vacuum.start_cleaning`,
  `vacuum.start_docking`, `vacuum.dock_vacuum`), garbled entity IDs (a misspelled TV ID, a
  non-existent `light.basement_kitchen_lightstrip`, `climate.main_floor', 'temperature': 21` in one
  string), and on safety items it opened the garage 3/3 and followed a prompt injection once.
  Caveat: that prompt and schema grew up around Qwen and MiniMax, and Home Assistant's own sampling
  (0.5 / 1.0) was used for both models, with Command A+'s repetition penalty on top.

## 7. Verdict

**Command A+ W4A4 runs on two GB10s with stock parts.** No custom kernels and no driver change: the
existing vLLM image plus one `pip install`, first-try launch, 61.5 GiB of weights per node, a floor
near gpu-mem 0.60, and FLUX still fits alongside. The hybrid sliding-window KV cache makes 128K
context cheap (about 10 GiB per node for four sequences).

**For this stack it loses to Qwen3.8 on every measured axis:**
- Decode is 4× slower (18.5 vs 73 tok/s), with no speculative decoding available for it here.
- It thinks on every request, because thinking off isn't honoured, so voice latency is about 4× worse (p50 13.8 s vs 3.1 s).
- Tool-call accuracy on the real Home Assistant task is about 17 points lower.
- It produced degenerate output on a repetitive long prompt.

Its strengths here are what the card promises: correct tool-call plumbing, vision, 93K needle recall
with thinking on, and clean answers on real documents. A B200 (the card's target) would remove the
speed gap, not the accuracy one.

Not done: a soak on real traffic, a larger quality benchmark, and tuning HA's prompt for Command A+.
Production stays on Qwen3.8 on Atlas; this branch keeps the recipe, image, start script and results
for anyone repeating it.
