"""Query understanding: condense follow-ups, and decompose only when it pays off.

Production pattern that keeps showing up: classic single-shot RAG as the default
(fast, cheap, good for "what is our PTO policy"), and a heavier path with query
decomposition / multiple retrieval rounds only for questions that need it
(comparisons, multi-hop "which services owned by team X use library Y").
A cheap router decides. Start with a heuristic like this one; replace with a
small classifier once you have logged real queries.
"""
from __future__ import annotations

import json
import re

from ragkit.llm import LLMClient

_COMPLEX_HINTS = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference between|both|each of|all of the|which .* and|"
    r"step[- ]by[- ]step|across|relationship between)\b",
    re.IGNORECASE,
)


def needs_decomposition(question: str) -> bool:
    return bool(_COMPLEX_HINTS.search(question)) or question.count("?") > 1


CONDENSE = """Rewrite the final user question so it can be understood without the conversation.
Keep names, ids and acronyms exactly as written. Output only the rewritten question.

Conversation:
{history}

Final question: {question}"""

DECOMPOSE = """Break the question into at most 3 standalone search queries that together
cover what is needed to answer it. Keep exact names, ids and acronyms.
Return a JSON list of strings and nothing else.

Question: {question}"""


def condense(history: list[dict], question: str, llm: LLMClient) -> str:
    if not history:
        return question
    h = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    return llm.complete([{"role": "user", "content": CONDENSE.format(history=h, question=question)}], 200).strip() or question


def decompose(question: str, llm: LLMClient) -> list[str]:
    raw = llm.complete([{"role": "user", "content": DECOMPOSE.format(question=question)}], 200)
    try:
        subs = [s for s in json.loads(raw[raw.find("[") : raw.rfind("]") + 1]) if isinstance(s, str)]
    except (ValueError, json.JSONDecodeError):
        subs = []
    return [question, *subs[:3]]
