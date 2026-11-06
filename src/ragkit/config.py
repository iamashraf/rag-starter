"""Settings loaded from environment variables (and an optional .env file).

Kept dependency-free on purpose. Swap for pydantic-settings if you prefer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _env(name: str, default: str) -> str:
    return os.environ.get(f"RAGKIT_{name}", default)


def _bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # LLM (OpenAI-compatible; Ollama works via http://localhost:11434/v1)
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.openai.com/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-4o-mini"))

    # Embeddings
    embed_provider: str = field(default_factory=lambda: _env("EMBED_PROVIDER", "hash"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "text-embedding-3-small"))
    embed_base_url: str = field(default_factory=lambda: _env("EMBED_BASE_URL", "https://api.openai.com/v1"))
    embed_api_key: str = field(default_factory=lambda: _env("EMBED_API_KEY", ""))
    embed_dim: int = field(default_factory=lambda: int(_env("EMBED_DIM", "384")))

    # Chunking
    chunk_target_tokens: int = field(default_factory=lambda: int(_env("CHUNK_TOKENS", "350")))
    chunk_overlap_tokens: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "50")))
    contextualize: bool = field(default_factory=lambda: _bool("CONTEXTUALIZE", False))

    # Retrieval: retrieve wide, rerank narrow
    candidates_per_retriever: int = field(default_factory=lambda: int(_env("CANDIDATES", "30")))
    rrf_k: int = field(default_factory=lambda: int(_env("RRF_K", "60")))
    final_top_k: int = field(default_factory=lambda: int(_env("TOP_K", "6")))
    reranker: str = field(default_factory=lambda: _env("RERANKER", "none"))
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "BAAI/bge-reranker-base"))
    # Below this fused/rerank score we treat retrieval as "nothing relevant" and refuse.
    min_relevance: float = field(default_factory=lambda: float(_env("MIN_RELEVANCE", "0.0")))

    # Storage + audit
    db_path: str = field(default_factory=lambda: _env("DB_PATH", "./data/ragkit.db"))
    audit_path: str = field(default_factory=lambda: _env("AUDIT_PATH", "./data/audit.jsonl"))


def get_settings() -> Settings:
    _load_dotenv()
    return Settings()
