"""HTTP API (needs the [api] extra):  uvicorn ragkit.api:app --reload

IMPORTANT: the principal must come from a verified identity (OIDC/JWT from your IdP
via your gateway), never from a field the caller controls. The header-based
`get_principal` below is a stub for local testing only.
"""
from __future__ import annotations

import os
from typing import Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ragkit.app import build_app
from ragkit.guardrails import GuardrailError
from ragkit.models import Principal

if os.environ.get("RAGKIT_INSECURE_HEADER_AUTH") != "1":
    raise RuntimeError(
        "api.py trusts x-user / x-groups headers, so anyone who can reach it can claim any group. "
        "Replace get_principal with JWT validation, or set RAGKIT_INSECURE_HEADER_AUTH=1 for local testing."
    )

app = FastAPI(title="ragkit")
_rag = build_app()


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class AskRequest(BaseModel):
    question: str
    history: list[Message] = Field(default=[], max_length=6)


def get_principal(x_user: str = Header(...), x_groups: str = Header("")) -> Principal:
    # TODO: replace with JWT validation, then map IdP groups to ACL tokens.
    return Principal(x_user, frozenset(g.strip() for g in x_groups.split(",") if g.strip()))


@app.post("/ask")
def ask(req: AskRequest, principal: Principal = Depends(get_principal)):
    try:
        ans = _rag.pipeline.ask(req.question, principal, [m.model_dump() for m in req.history])
    except GuardrailError as e:
        raise HTTPException(400, str(e)) from e
    except httpx.HTTPError as e:
        raise HTTPException(502, "model provider error") from e
    return {
        "answer": ans.text,
        "refused": ans.refused,
        "trace_id": ans.trace_id,
        "sources": [{"id": i, "title": c.title, "uri": c.uri, "section": c.chunk.heading_path}
                    for i, c in enumerate(ans.citations, 1)],
    }


@app.get("/healthz")
def health():
    return {"ok": True, **_rag.store.stats()}
