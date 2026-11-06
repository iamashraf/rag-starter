"""Wiring: build all components from settings in one place."""
from __future__ import annotations

from dataclasses import dataclass

from ragkit.audit import AuditLog
from ragkit.config import Settings, get_settings
from ragkit.llm import Embedder, LLMClient, build_embedder, build_llm
from ragkit.pipeline import RAGPipeline
from ragkit.retrieval import HybridRetriever, build_reranker
from ragkit.store import Store


@dataclass
class App:
    settings: Settings
    store: Store
    embedder: Embedder
    llm: LLMClient | None
    pipeline: RAGPipeline


def build_app(settings: Settings | None = None, use_llm: bool = True) -> App:
    s = settings or get_settings()
    store = Store(s.db_path)
    embedder = build_embedder(s)
    llm = build_llm(s) if (use_llm and s.llm_api_key) else None
    retriever = HybridRetriever(store, embedder, s, build_reranker(s))
    return App(s, store, embedder, llm, RAGPipeline(retriever, s, llm, AuditLog(s.audit_path)))
