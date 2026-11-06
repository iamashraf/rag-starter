"""Layered guardrails. Each layer is independent so one failing does not open the whole system.

  input      : validate the user's question (size, obvious injection attempts)
  retrieval  : ACL pre-filter + freshness live in store.py / retrieval.py
  context    : treat retrieved chunks as UNTRUSTED data; scan them, fence them
  generation : system prompt that answers only from sources and cites them
  output     : verify citations point at real sources; redact sensitive patterns
  action     : (for agentic use) allow-list tools; never let retrieved text trigger writes

Regex scanners are a cheap first line, not a complete defense. Indirect prompt
injection has no equivalent of parameterized SQL. Plan to add a classifier
(Llama Guard / Prompt Guard, a managed guardrail service, or an LLM judge) behind
the same function signatures.
"""
from __future__ import annotations

import html
import re

from ragkit.models import RetrievedChunk

MAX_QUESTION_CHARS = 2000

_INJECTION_PATTERNS = {
    "ignore_instructions": r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|all)\b.{0,20}\b(instructions?|rules?|prompts?)\b",
    "role_override": r"\byou are now\b|\bact as (an?|the) (system|admin|developer)\b|\bnew (system )?instructions?\s*:",
    "system_tag": r"<\s*/?\s*(system|assistant|instructions?)\s*>|\[\s*system\s*\]",
    "exfiltration": r"\b(send|post|upload|exfiltrate|forward)\b.{0,60}\b(https?://|webhook|endpoint|email address)",
    "hidden_ai_address": r"\bif you are an? (ai|llm|language model|assistant)\b",
    "secret_probe": r"\b(reveal|print|show)\b.{0,30}\b(system prompt|api key|password|secret)s?\b",
}
_COMPILED = {k: re.compile(v, re.IGNORECASE | re.DOTALL) for k, v in _INJECTION_PATTERNS.items()}

_REDACT = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "card": re.compile(r"\b\d(?:[ -]?\d){12,15}\b"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}


def scan_for_injection(text: str) -> list[str]:
    return [name for name, rx in _COMPILED.items() if rx.search(text)]


class GuardrailError(ValueError):
    pass


# ---------- input ----------
def check_input(question: str) -> list[str]:
    q = question.strip()
    if not q:
        raise GuardrailError("Empty question")
    if len(q) > MAX_QUESTION_CHARS:
        raise GuardrailError(f"Question longer than {MAX_QUESTION_CHARS} characters")
    # Do not block on these by default (false positives annoy users); record them for review.
    return [f"input:{h}" for h in scan_for_injection(q)]


# ---------- context ----------
def screen_context(items: list[RetrievedChunk], drop_flagged: bool = True) -> tuple[list[RetrievedChunk], list[str]]:
    kept, events = [], []
    for it in items:
        # heading_path carries the doc title and headings, which also reach the model.
        hits = scan_for_injection(f"{it.chunk.heading_path}\n{it.chunk.text}")
        if hits:
            events.append(f"context:{it.chunk.doc_id}:{'|'.join(hits)}")
            if drop_flagged:
                continue
        kept.append(it)
    return kept, events


_SOURCE_TAG = re.compile(r"<\s*(/?)\s*source\b", re.IGNORECASE)


def fence(i: int, it: RetrievedChunk) -> str:
    # Neutralize anything that could open or close our fence, in the body and in the attributes.
    body = _SOURCE_TAG.sub(r"<\1 source", it.chunk.text)
    title, section = html.escape(it.title), html.escape(it.chunk.heading_path)
    return f'<source id="{i}" title="{title}" section="{section}">\n{body}\n</source>'


# ---------- output ----------
_CITE = re.compile(r"\[(\d+)\]")


def check_output(answer: str, n_sources: int) -> tuple[str, list[str]]:
    events: list[str] = []
    cited = {int(m) for m in _CITE.findall(answer)}
    bad = sorted(c for c in cited if c < 1 or c > n_sources)
    if bad:
        events.append(f"output:invalid_citations:{bad}")
    if not cited and not is_refusal(answer):
        events.append("output:no_citations")
    for name, rx in _REDACT.items():
        redacted = rx.sub(lambda m: f"[redacted {name}]" if name != "card" or _luhn(m.group()) else m.group(), answer)
        if redacted != answer:
            answer = redacted
            events.append(f"output:redacted:{name}")
    return answer, events


REFUSAL = "I could not find this in the documents you have access to."


def _luhn(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()][::-1]
    return sum(d if i % 2 == 0 else sum(divmod(d * 2, 10)) for i, d in enumerate(digits)) % 10 == 0


def _norm(text: str) -> str:
    return text.strip().strip("\"'").rstrip(".").lower()


def is_refusal(text: str) -> bool:
    return _norm(text) == _norm(REFUSAL)
