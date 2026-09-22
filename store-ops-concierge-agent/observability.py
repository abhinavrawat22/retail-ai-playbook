"""
Observability layer: a LangChain callback handler that captures tool calls,
LLM calls, latency, approximate token/cost usage, and exposes it as a list
of structured trace events the Streamlit UI can render live.

For a production setup, swap this handler for LangSmith
(https://smith.langchain.com) or an OpenTelemetry exporter - the callback
interface is the same either way.
"""

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler

# Rough, illustrative pricing (USD per 1K tokens) - NOT real-time pricing.
# Good enough for teaching cost-awareness in a lab, not for billing.
INPUT_COST_PER_1K = 0.00015
OUTPUT_COST_PER_1K = 0.0006


def estimate_tokens(text: str) -> int:
    """Very rough token estimate (~4 chars/token) to avoid a tokenizer dependency."""
    return max(1, len(text) // 4)


@dataclass
class TraceEvent:
    kind: str  # "llm" | "tool" | "guardrail"
    name: str
    detail: str
    latency_ms: float
    agent_name: str = "unknown_agent"
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ObservabilityHandler(BaseCallbackHandler):
    """Collects trace events for one agent invocation."""

    def __init__(self, agent_name: str = "Store Ops Concierge"):
        self.agent_name = agent_name
        self.events: list[TraceEvent] = []
        self._starts: dict[str, float] = {}

    # --- LLM lifecycle -----------------------------------------------------
    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id: UUID, **kwargs: Any) -> None:
        self._starts[str(run_id)] = time.time()

    def on_llm_end(self, response, *, run_id: UUID, **kwargs: Any) -> None:
        start = self._starts.pop(str(run_id), time.time())
        latency_ms = (time.time() - start) * 1000
        text = ""
        try:
            text = response.generations[0][0].text
        except Exception:
            pass
        tokens_out = estimate_tokens(text)
        tokens_in = estimate_tokens(str(response))
        cost = (tokens_in / 1000 * INPUT_COST_PER_1K) + (tokens_out / 1000 * OUTPUT_COST_PER_1K)
        self.events.append(
            TraceEvent(
                kind="llm",
                name="chat_model_call",
                detail=text[:200],
                latency_ms=latency_ms,
                agent_name=self.agent_name,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
            )
        )

    # --- Tool lifecycle ------------------------------------------------------
    def on_tool_start(self, serialized: dict, input_str: str, *, run_id: UUID, **kwargs: Any) -> None:
        self._starts[str(run_id)] = time.time()
        self._pending_tool_name = serialized.get("name", "unknown_tool")
        self._pending_input = input_str

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        start = self._starts.pop(str(run_id), time.time())
        latency_ms = (time.time() - start) * 1000
        self.events.append(
            TraceEvent(
                kind="tool",
                name=getattr(self, "_pending_tool_name", "unknown_tool"),
                detail=f"input={getattr(self, '_pending_input', '')!r} -> output={str(output)[:200]!r}",
                latency_ms=latency_ms,
                agent_name=self.agent_name,
            )
        )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        start = self._starts.pop(str(run_id), time.time())
        latency_ms = (time.time() - start) * 1000
        self.events.append(
            TraceEvent(
                kind="tool",
                name=getattr(self, "_pending_tool_name", "unknown_tool"),
                detail=f"ERROR: {error}",
                latency_ms=latency_ms,
                agent_name=self.agent_name,
            )
        )

    def add_guardrail_event(self, name: str, detail: str) -> None:
        self.events.append(
            TraceEvent(kind="guardrail", name=name, detail=detail, latency_ms=0.0, agent_name=self.agent_name)
        )

    def tool_call_summaries(self) -> list[dict]:
        """Returns a compact list of {agent_name, tool_name, detail} for every
        tool call made during this invocation - used to answer 'which tool
        call was used by which agent'."""
        return [
            {"agent_name": e.agent_name, "tool_name": e.name, "detail": e.detail}
            for e in self.events
            if e.kind == "tool"
        ]

    def totals(self) -> dict:
        return {
            "total_latency_ms": sum(e.latency_ms for e in self.events),
            "total_tokens_in": sum(e.tokens_in for e in self.events),
            "total_tokens_out": sum(e.tokens_out for e in self.events),
            "total_cost_usd": round(sum(e.cost_usd for e in self.events), 6),
            "tool_calls": sum(1 for e in self.events if e.kind == "tool"),
        }
