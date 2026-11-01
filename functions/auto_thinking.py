"""
title: Auto Thinking
author: ai-stack
description: >
  On the thinking model, skips the hidden reasoning step for messages that clearly don't need it
  (greetings, thanks, short chit-chat and simple requests), so voice mode answers faster. Anything
  that looks like real reasoning (why/how/explain/compare/plan/code/math, or a long message) still
  thinks, and so does every message in a chat with an external tool server switched on (e.g. Google):
  without thinking, the model tends to overlook the tools. Start a message with "think:" to force
  thinking, or "quick:" to force a fast reply.
version: 0.2.0
"""
import re
from typing import Optional
from pydantic import BaseModel, Field

GREETING = re.compile(
    r"^\s*(hi|hey|hello|yo|good (morning|afternoon|evening|night)|thanks?( you)?|thank you|ok(ay)?|"
    r"cool|great|nice|awesome|got it|sounds good|yes|yeah|yep|no|nope|bye|goodbye|see you|"
    r"how are you|what'?s up)\b[\s!.,?]*$",
    re.I,
)
NEEDS_THINKING = re.compile(
    r"\b(why|how (do|does|did|can|could|would|should|much|many)|explain|compare|difference|versus|vs\.?|"
    r"plan|analy[sz]e|calculate|compute|solve|prove|derive|estimate|debug|fix|code|script|function|"
    r"program|algorithm|design|architect|recommend|should i|pros and cons|trade-?offs?|step[- ]by[- ]step|"
    r"summari[sz]e|review|evaluate|research|translate|write (a|an|me)|draft)\b",
    re.I,
)
MATHY = re.compile(r"[=<>^*/%]|\d+\s*[-+x×]\s*\d+|```|\d\s*(times|plus|minus|divided|over|percent|squared)\b|\b(square root|how much|how many)\b", re.I)


class Filter:
    class Valves(BaseModel):
        model_ids: str = Field(
            default="Qwen3.8-Flash-Next-NVFP4-Trial",
            description="Comma-separated model ids this filter acts on (the thinking variants).",
        )
        max_words_fast: int = Field(
            default=12, description="Messages longer than this always think."
        )
        think_with_tool_servers: bool = Field(
            default=True,
            description="Keep thinking on when the chat has an external tool server (MCP/OpenAPI, e.g. Google) "
            "switched on. Without it the model often answers 'I have no such tool'.",
        )
        show_status: bool = Field(
            default=False, description="Show a small 'quick reply' status line when thinking is skipped."
        )

    def __init__(self):
        self.valves = self.Valves()

    def decide(self, text: str) -> tuple[bool, str]:
        """Returns (skip_thinking, reason)."""
        t = text.strip()
        low = t.lower()
        if low.startswith("think:"):
            return False, "forced"
        if low.startswith("quick:"):
            return True, "forced"
        if GREETING.match(t):
            return True, "greeting/ack"
        if len(t.split()) > self.valves.max_words_fast:
            return False, "long"
        if NEEDS_THINKING.search(t) or MATHY.search(t):
            return False, "needs reasoning"
        return True, "short/simple"

    async def inlet(self, body: dict, __event_emitter__=None, __user__: Optional[dict] = None) -> dict:
        ids = {m.strip() for m in self.valves.model_ids.split(",") if m.strip()}
        if body.get("model") not in ids:
            return body
        msgs = body.get("messages") or []
        last = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
        if not last:
            return body
        content = last.get("content")
        if isinstance(content, list):   # multimodal: images need the full model; keep thinking
            if any(p.get("type") != "text" for p in content):
                return body
            content = " ".join(p.get("text", "") for p in content)
        skip, reason = self.decide(content or "")
        # body["tool_ids"] is still present at inlet time (OpenWebUI pops it after the filters run).
        if skip and reason != "forced" and self.valves.think_with_tool_servers and any(
            str(t).startswith("server:") for t in (body.get("tool_ids") or [])
        ):
            skip = False
        if skip:
            # Same switch the "-Fast" alias uses (LiteLLM forwards it to SGLang's chat template).
            body["chat_template_kwargs"] = {**(body.get("chat_template_kwargs") or {}), "enable_thinking": False}
            if self.valves.show_status and __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": "Quick reply (no thinking)", "done": True}})
        return body
