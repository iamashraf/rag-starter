"""Shared helpers for the pattern demos."""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

import httpx

from ragkit.config import Settings
from ragkit.ingest.loaders import LocalFolderLoader
from ragkit.ingest.sync import sync
from ragkit.llm import build_embedder
from ragkit.retrieval import HybridRetriever
from ragkit.store import Store

OLLAMA_TAGS = "http://localhost:11434/api/tags"


class Unavailable(Exception):
    """The environment cannot run this demo."""


class ScriptedLLM:
    def __init__(self, answer: str):
        self.answer = answer

    def complete(self, messages: list[dict], max_tokens: int = 800, temperature: float = 0.0) -> str:
        return self.answer


def require_ollama(model_names: list[str]) -> None:
    try:
        r = httpx.get(OLLAMA_TAGS, timeout=5)
        r.raise_for_status()
        have = [m["name"] for m in r.json().get("models", [])]
    except (httpx.HTTPError, ValueError) as e:
        raise Unavailable(
            f"Ollama is not reachable at {OLLAMA_TAGS} ({e}). "
            "Start it with: podman compose -f compose.ollama.yaml up -d"
        ) from e
    for name in model_names:
        want = name if ":" in name else f"{name}:latest"
        if want in have:
            continue
        tagged = [h for h in have if h.split(":")[0] == name.split(":")[0]] if ":" not in name else []
        hint = f" (found {', '.join(tagged)}, but a bare name matches only :latest)" if tagged else ""
        fix = f"set the model to the tagged name (for example {tagged[0]})" if tagged else f"run: ollama pull {name}"
        raise Unavailable(f"Ollama model '{name}' is missing{hint}. To fix, {fix}")


def make_settings(tmp: Path, embedder: str = "hash") -> Settings:
    s = Settings()
    s.db_path = str(tmp / "demo.db")
    s.audit_path = str(tmp / "audit.jsonl")
    s.reranker = "none"
    s.min_relevance = 0.0
    s.contextualize = False
    if embedder == "ollama":
        s.embed_provider = "ollama"
        s.embed_model = os.environ.get("RAGKIT_EMBED_MODEL", "nomic-embed-text")
        require_ollama([s.embed_model])
    else:
        s.embed_provider = "hash"
    return s


def build(tmp: Path, docs: Path, embedder: str = "hash", llm=None):
    s = make_settings(tmp, embedder)
    store = Store(s.db_path)
    emb = build_embedder(s)
    rep = sync(LocalFolderLoader(docs).load(), store, emb, s, source="local", llm=llm)
    return s, store, emb, HybridRetriever(store, emb, s), rep


def report(name: str, expected: str, observed: str, ok: bool) -> bool:
    print(f"[{name}] EXPECTED: {expected}")
    print(f"[{name}] OBSERVED: {observed}")
    print(f"[{name}] {'PASS' if ok else 'FAIL'}")
    return ok


def cli(main: Callable[[str], bool], default: str = "hash") -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ollama", action="store_true", help="use local Ollama embeddings")
    args = p.parse_args()
    try:
        ok = main("ollama" if args.ollama else default)
    except Unavailable as e:
        print(f"UNAVAILABLE: {e}")
        sys.exit(2)
    sys.exit(0 if ok else 1)
