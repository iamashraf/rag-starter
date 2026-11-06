from pathlib import Path

import pytest

from ragkit.audit import AuditLog
from ragkit.config import Settings
from ragkit.ingest.loaders import LocalFolderLoader
from ragkit.ingest.sync import sync
from ragkit.llm import HashEmbedder
from ragkit.pipeline import RAGPipeline
from ragkit.retrieval import HybridRetriever
from ragkit.store import Store

SAMPLE = Path(__file__).resolve().parent.parent / "sample_docs"


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.db_path = str(tmp_path / "t.db")
    s.audit_path = str(tmp_path / "audit.jsonl")
    s.embed_provider = "hash"
    s.reranker = "none"
    return s


@pytest.fixture
def indexed(settings):
    store = Store(settings.db_path)
    emb = HashEmbedder(settings.embed_dim)
    report = sync(LocalFolderLoader(SAMPLE).load(), store, emb, settings, source="local")
    retriever = HybridRetriever(store, emb, settings)
    pipe = RAGPipeline(retriever, settings, None, AuditLog(settings.audit_path))
    return store, emb, retriever, pipe, report
