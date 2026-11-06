"""Minimal OpenAI-compatible chat + embeddings client over httpx.

Works with OpenAI, Azure OpenAI (via compatible gateway), vLLM, LiteLLM, and Ollama
(base_url http://localhost:11434/v1, any api_key). Keeping it this thin means you
can point it at an internal model gateway without pulling a vendor SDK.
"""
from __future__ import annotations

import hashlib

import httpx
import numpy as np

from ragkit.config import Settings


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120):
        self.model = model
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key or 'none'}"},
            timeout=timeout,
        )

    def complete(self, messages: list[dict], max_tokens: int = 800, temperature: float = 0.0) -> str:
        r = self._http.post(
            "/chat/completions",
            json={"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


class Embedder:
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:  # (n, dim), L2-normalized
        raise NotImplementedError


class HashEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder. NOT semantic. For tests and offline demos only."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                tok = "".join(ch for ch in tok if ch.isalnum())
                if not tok:
                    continue
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        return _normalize(out)


class OpenAICompatEmbedder(Embedder):
    def __init__(self, base_url: str, api_key: str, model: str, batch: int = 64):
        self.model, self.batch = model, batch
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), headers={"Authorization": f"Bearer {api_key or 'none'}"}, timeout=120
        )
        self.dim = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            r = self._http.post("/embeddings", json={"model": self.model, "input": texts[i : i + self.batch]})
            r.raise_for_status()
            vecs.extend(d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"]))
        arr = np.asarray(vecs, dtype=np.float32)
        self.dim = arr.shape[1] if arr.size else self.dim
        return _normalize(arr)


def _normalize(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


def build_embedder(s: Settings) -> Embedder:
    if s.embed_provider == "hash":
        return HashEmbedder(s.embed_dim)
    if s.embed_provider == "ollama":
        return OpenAICompatEmbedder("http://localhost:11434/v1", "ollama", s.embed_model)
    return OpenAICompatEmbedder(s.embed_base_url, s.embed_api_key, s.embed_model)


def build_llm(s: Settings) -> LLMClient:
    return LLMClient(s.llm_base_url, s.llm_api_key, s.llm_model)
