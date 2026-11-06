"""Append-only audit log: who asked what, which chunks were retrieved, what was answered.

Security reviews will ask "prove this user was allowed to see the source behind that
answer". This log answers it. It is also your best debugging tool: most bad answers
are retrieval problems, and you can only see that if you log what was retrieved.
Ship these lines to your SIEM / log pipeline (and to OTel traces) in production.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from ragkit.models import Answer, Principal


class AuditLog:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, principal: Principal, question: str, queries: list[str], answer: Answer, latency_ms: float) -> str:
        trace_id = answer.trace_id or uuid.uuid4().hex
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trace_id": trace_id,
            "user": principal.user_id,
            "groups": sorted(principal.groups),
            "question": question,
            "queries": queries,
            "retrieved": [
                {"chunk_id": c.chunk.chunk_id, "doc_id": c.chunk.doc_id, "acl": c.chunk.acl,
                 "score": round(c.score, 5), "signals": c.signals}
                for c in answer.retrieved
            ],
            "refused": answer.refused,
            "guardrail_events": answer.guardrail_events,
            "latency_ms": round(latency_ms, 1),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return trace_id
