"""
Audit and metrics logging.

Two separate JSON-lines log files under logs/:
- audit.log   -> one record per notable activity (user query received, each
                 guardrail decision, each tool call, final answer returned,
                 rate-limit blocks). This is the "who did what, when, why"
                 trail a compliance/security reviewer would want.
- metrics.log -> one aggregated record per completed turn (latency, token
                 counts, estimated cost, tool-call counts, guardrail-fired
                 counts). This is the "how is the system performing" trail
                 an SRE/ops dashboard would consume.

Kept dependency-free (stdlib `logging` + `json`) so the lab has zero extra
setup. In production, swap the FileHandlers below for a structured logging
sink (e.g. an OpenTelemetry log exporter, CloudWatch, or a SIEM).
"""

import json
import logging
import time
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def _make_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(LOG_DIR / filename, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


audit_logger = _make_logger("store_ops.audit", "audit.log")
metrics_logger = _make_logger("store_ops.metrics", "metrics.log")


def _now() -> dict:
    return {"timestamp": time.time(), "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def log_audit_event(event_type: str, session_id: str, agent_name: str, details: dict) -> None:
    """Append one structured audit record. event_type examples:
    'user_query_received', 'guardrail_decision', 'tool_call', 'agent_response',
    'rate_limit_blocked'."""
    record = {
        **_now(),
        "event_type": event_type,
        "session_id": session_id,
        "agent_name": agent_name,
        **details,
    }
    audit_logger.info(json.dumps(record, default=str))


def log_metrics_event(session_id: str, agent_name: str, totals: dict, tool_call_names: list[str],
                       guardrails_fired: list[str]) -> None:
    """Append one aggregated metrics record for a completed turn."""
    record = {
        **_now(),
        "session_id": session_id,
        "agent_name": agent_name,
        "latency_ms": totals.get("total_latency_ms", 0),
        "tokens_in": totals.get("total_tokens_in", 0),
        "tokens_out": totals.get("total_tokens_out", 0),
        "cost_usd": totals.get("total_cost_usd", 0),
        "tool_call_count": totals.get("tool_calls", 0),
        "tool_calls": tool_call_names,
        "guardrails_fired": guardrails_fired,
    }
    metrics_logger.info(json.dumps(record, default=str))
