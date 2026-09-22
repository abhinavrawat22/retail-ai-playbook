"""
Guardrails layer: input filtering (prompt injection, jailbreak, PII) and
output filtering (PII redaction, policy enforcement) plus a simple
in-memory rate limiter.

These are deliberately implemented with plain regex/heuristics (no external
guardrail service) to keep the lab dependency-free. The README explains
where you would swap in a production-grade guardrail service.
"""

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# PII detection / redaction
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Redact emails, phone numbers, and credit-card-like numbers from text.
    Returns (redacted_text, list_of_redaction_types_applied)."""
    redactions = []
    if EMAIL_RE.search(text):
        text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
        redactions.append("email")
    if PHONE_RE.search(text):
        text = PHONE_RE.sub("[REDACTED_PHONE]", text)
        redactions.append("phone")
    if CREDIT_CARD_RE.search(text):
        text = CREDIT_CARD_RE.sub("[REDACTED_CARD]", text)
        redactions.append("card")
    return text, redactions


# ---------------------------------------------------------------------------
# Prompt injection / jailbreak heuristics
# ---------------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore (all |any |previous |prior )*(instructions|prompt|rules)",
    r"disregard (your|the) (system|previous) prompt",
    r"reveal (your|the) (system|hidden) prompt",
    r"you are now (in )?(dan|developer|jailbreak) mode",
    r"act as if you have no (restrictions|guardrails|rules)",
    r"print your (instructions|system prompt|configuration)",
    r"override (your|the) (rules|policy|guardrails)",
    r"ignore (the )?(discount|policy)( policy)? (cap|limit|ceiling)",
    r"bypass (the )?(discount|policy)( policy)? (cap|limit|ceiling)",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def detect_prompt_injection(text: str) -> bool:
    return bool(INJECTION_RE.search(text))


# ---------------------------------------------------------------------------
# Input guard
# ---------------------------------------------------------------------------
@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""
    sanitized_text: str = ""
    events: list[str] = field(default_factory=list)


def input_guard(user_input: str) -> GuardResult:
    """Runs before the agent sees the user's message."""
    events = []

    if detect_prompt_injection(user_input):
        events.append("prompt_injection_detected")
        return GuardResult(
            allowed=False,
            reason="Your message looks like it is trying to override the assistant's "
                   "instructions or safety rules, so it was blocked.",
            sanitized_text=user_input,
            events=events,
        )

    sanitized, redactions = redact_pii(user_input)
    if redactions:
        events.append(f"input_pii_redacted:{','.join(redactions)}")

    return GuardResult(allowed=True, sanitized_text=sanitized, events=events)


# ---------------------------------------------------------------------------
# Output guard
# ---------------------------------------------------------------------------
def output_guard(agent_response: str) -> GuardResult:
    """Runs on the agent's final answer before it is shown to the user."""
    events = []
    sanitized, redactions = redact_pii(agent_response)
    if redactions:
        events.append(f"output_pii_redacted:{','.join(redactions)}")
    return GuardResult(allowed=True, sanitized_text=sanitized, events=events)


# ---------------------------------------------------------------------------
# Rate limiter (simple in-memory sliding window, per session)
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, session_id: str) -> tuple[bool, int]:
        """Returns (allowed, remaining_requests_in_window)."""
        now = time.time()
        window = self._hits[session_id]
        while window and now - window[0] > self.window_seconds:
            window.popleft()
        if len(window) >= self.max_requests:
            return False, 0
        window.append(now)
        return True, self.max_requests - len(window)
