"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion, then optional rerank.

Why hybrid for internal knowledge: employees search with ticket ids, service names,
acronyms, error codes and policy numbers. Embeddings blur those; BM25 nails them.
Embeddings catch paraphrase; BM25 misses it. Most real queries need both.

Why RRF: BM25 scores and cosine similarities live on different scales. RRF ignores
raw scores and uses only rank positions, so there is nothing to normalize or tune
beyond k (60 is the usual default). It is the built-in fusion in Elastic, OpenSearch,
Weaviate, Qdrant and Azure AI Search.

Why rerank: bi-encoder embeddings compress a chunk into one vector. A cross-encoder
reads query and chunk together and is far more precise, but too slow for the whole
corpus. So: retrieve wide (30 + 30), fuse, rerank narrow (top 6).
"""
from __future__ import annotations

import json
from collections import defaultdict

from ragkit.config import Settings
from ragkit.llm import Embedder
from ragkit.models import Chunk, Principal, RetrievedChunk
from ragkit.store import Store


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] += 1.0 / (k + rank)
    return dict(scores)


class Reranker:
    def rerank(self, query: str, items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return items


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # [rerank] extra

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not items:
            return items
        scores = self.model.predict([(query, it.chunk.embed_text) for it in items])
        for it, s in zip(items, scores, strict=True):
            it.signals["rerank"] = float(s)
            it.score = float(s)
        return sorted(items, key=lambda x: x.score, reverse=True)


def build_reranker(s: Settings) -> Reranker:
    return CrossEncoderReranker(s.rerank_model) if s.reranker == "cross-encoder" else Reranker()


class HybridRetriever:
    def __init__(self, store: Store, embedder: Embedder, settings: Settings, reranker: Reranker | None = None):
        self.store, self.embedder, self.s = store, embedder, settings
        store.bind_embedder(embedder.name)
        self.reranker = reranker or Reranker()

    def retrieve(self, queries: list[str], principal: Principal) -> list[RetrievedChunk]:
        """Accepts one or more queries (original + rewrites/sub-questions) and fuses them all."""
        rankings: list[list[str]] = []
        signals: dict[str, dict[str, float]] = defaultdict(dict)
        tokens = principal.acl_tokens
        n = self.s.candidates_per_retriever
        for q in queries:
            bm25 = [c for c, _ in self.store.search_bm25(q, tokens, n)]
            dense = [c for c, _ in self.store.search_dense(self.embedder.embed([q])[0], tokens, n)]
            rankings += [bm25, dense]
            for r, cid in enumerate(bm25, 1):
                signals[cid].setdefault("bm25_rank", r)
            for r, cid in enumerate(dense, 1):
                signals[cid].setdefault("dense_rank", r)

        fused = rrf(rankings, self.s.rrf_k)
        top_ids = sorted(fused, key=fused.get, reverse=True)[: max(self.s.final_top_k * 4, 20)]
        rows = self.store.fetch(top_ids)

        items: list[RetrievedChunk] = []
        for cid in top_ids:
            row = rows.get(cid)
            if not row or row["status"] != "active":
                continue  # freshness check at read time: never serve a tombstoned doc
            # Defense in depth: re-check ACL on the fetched row, even though the search already filtered.
            if not set(json.loads(row["acl"])) & tokens:
                continue
            chunk = Chunk(
                chunk_id=cid, doc_id=row["doc_id"], ordinal=row["ordinal"], text=row["text"],
                embed_text=row["embed_text"], heading_path=row["heading_path"] or "",
                acl=json.loads(row["acl"]), metadata=json.loads(row["metadata"] or "{}"),
            )
            items.append(RetrievedChunk(chunk, row["title"], row["uri"], fused[cid], signals[cid] | {"rrf": fused[cid]}))

        items = self.reranker.rerank(queries[0], items)
        return items[: self.s.final_top_k]
