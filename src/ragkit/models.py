"""Core data types. Everything else passes these around."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Principal:
    """The person asking. Resolve this from your IdP (Okta/Entra/LDAP), never from the request body."""

    user_id: str
    groups: frozenset[str] = frozenset()

    @property
    def acl_tokens(self) -> set[str]:
        # A chunk is visible if it shares at least one token with the principal.
        # "public" is a convenience token for docs everyone may read.
        return {f"user:{self.user_id}", "public", *(f"group:{g}" for g in self.groups)}


@dataclass
class SourceDocument:
    """One document as it exists in the source system (Confluence page, Git file, SharePoint item...)."""

    doc_id: str  # stable id from the source system, NOT a file path you might rename
    title: str
    text: str
    source: str  # e.g. "confluence", "git", "local"
    uri: str  # link back to the original, used in citations
    acl: list[str]  # e.g. ["group:engineering", "user:alice"]
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: str | None = None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    text: str  # what the LLM sees
    embed_text: str  # what gets embedded and BM25-indexed (may include breadcrumb + LLM context)
    heading_path: str
    acl: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    title: str
    uri: str
    score: float
    signals: dict[str, float] = field(default_factory=dict)  # bm25_rank, dense_rank, rrf, rerank


@dataclass
class Answer:
    text: str
    citations: list[RetrievedChunk]
    refused: bool = False
    guardrail_events: list[str] = field(default_factory=list)
    trace_id: str = ""
    retrieved: list[RetrievedChunk] = field(default_factory=list)  # everything retrieved, cited or not
