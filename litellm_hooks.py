"""LiteLLM proxy hooks (loaded via litellm_settings.callbacks in litellm-config.yaml).

TaskDelay: requests for the "...-Task" alias wait TASK_DELAY_SECONDS (default 4) before going to the
model. OpenWebUI's background jobs (title, tags, follow-up suggestions) and Auto Memory use that
alias, and they all fire the moment a reply finishes, which in voice mode is exactly when Chatterbox
starts voicing the first sentence on worker, whose GPU also runs half of the LLM. Measured
2026-10-11: first-sentence TTS 1.64 s with the jobs running alongside, 0.87 s with them delayed 4 s.
The jobs' results (chips, tags, memories) just appear a few seconds later.

HAAutoThinking: Home Assistant (extended_openai_conversation on homeassistant) calls the "-Trial"
alias here directly, so OpenWebUI's Auto Thinking filter never sees its requests, and every voice
request used to think, even "turn off the kitchen lights". This applies the same decision to HA's
requests, recognised by HA's `execute_services` tool: quick commands and simple questions skip
thinking, why/how/explain/math or long requests still think. The rules are the filter's own
(functions/auto_thinking.py, mounted read-only as /app/auto_thinking.py), minus its "keep thinking
when tools are on" rule, since every HA request carries a tool. Each decision is logged.
OFF unless HA_AUTO_THINKING=1 (litellm env). Turned off 2026-10-24 after HA got one-word replies: the
cause was Atlas's loop detector (fixed with ATLAS_LOOP_NO_SUPPRESS in the Atlas recipe), but HA's
conversation still held the one-word turns, and with thinking off the model copies them. Re-enable
once HA's conversation has been reset, then check `docker logs litellm | grep ha-auto`.
"""
import asyncio
import os
import sys

from litellm.integrations.custom_logger import CustomLogger

DELAYED_SUFFIX = "-Task"


class TaskDelay(CustomLogger):
    def __init__(self):
        super().__init__()
        self.delay = float(os.environ.get("TASK_DELAY_SECONDS", "4"))

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if str(data.get("model", "")).endswith(DELAYED_SUFFIX) and self.delay > 0:
            await asyncio.sleep(self.delay)
        return data


task_delay = TaskDelay()


HA_TOOL = "execute_services"
HA_MODELS = {"Qwen3.8-Flash-Next-NVFP4-Trial"}


class HAAutoThinking(CustomLogger):
    def __init__(self):
        super().__init__()
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from auto_thinking import Filter
            self.decide = Filter().decide
        except Exception as e:  # missing mount: leave HA requests untouched rather than fail them
            print(f"[ha-auto-thinking] disabled, can't load auto_thinking.py: {e}", flush=True)
            self.decide = None

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if os.environ.get("HA_AUTO_THINKING", "0") != "1":  # off by default; see the module docstring
            return data
        if self.decide is None or data.get("model") not in HA_MODELS:
            return data
        if not any((t.get("function") or {}).get("name") == HA_TOOL for t in data.get("tools") or []):
            return data
        last = next((m for m in reversed(data.get("messages") or []) if m.get("role") == "user"), None)
        content = (last or {}).get("content") or ""
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        skip, reason = self.decide(content)
        if skip:
            data["chat_template_kwargs"] = {**(data.get("chat_template_kwargs") or {}), "enable_thinking": False}
        print(f"[ha-auto-thinking] {'fast' if skip else 'think'} ({reason}): {content[:60]!r}", flush=True)
        return data


ha_auto_thinking = HAAutoThinking()
