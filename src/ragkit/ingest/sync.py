"""Incremental sync: only re-embed what changed, and actually delete what was removed.

Indexes rot in two ways: edits that never get re-embedded, and deletes that never
propagate (ghost chunks keep answering with a retired policy). This module keeps a
content-hash ledger per document:

    new doc          -> chunk, embed, insert
    hash changed     -> replace all chunks for that doc atomically
    hash unchanged   -> skip (no embedding cost)
    gone from source -> tombstone, then delete chunks

The hash covers text + title + ACL, so a permission change re-indexes the doc's ACL
rows too. For large corpora you can split that into a cheaper ACL-only update path.

Run this from a scheduler (nightly full + frequent incremental) or from source
webhooks / CDC. Periodically run a full reconcile, because event streams miss things.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable

from ragkit.config import Settings
from ragkit.ingest.chunking import chunk_document
from ragkit.llm import Embedder, LLMClient
from ragkit.models import SourceDocument
from ragkit.store import Store


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    flagged: list[str] = field(default_factory=list)  # docs with suspected prompt injection
    chunks_embedded: int = 0

    def summary(self) -> str:
        return (f"added={len(self.added)} updated={len(self.updated)} skipped={len(self.skipped)} "
                f"deleted={len(self.deleted)} flagged={len(self.flagged)} chunks_embedded={self.chunks_embedded}")


def content_hash(doc: SourceDocument) -> str:
    payload = json.dumps({"t": doc.title, "x": doc.text, "a": sorted(doc.acl)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def sync(
    docs: Iterable[SourceDocument],
    store: Store,
    embedder: Embedder,
    settings: Settings,
    source: str,
    full_reconcile: bool = True,
    llm: LLMClient | None = None,
) -> SyncReport:
    from ragkit.guardrails import scan_for_injection  # local import to avoid a cycle

    report = SyncReport()
    seen: set[str] = set()

    for doc in docs:
        seen.add(doc.doc_id)
        h = content_hash(doc)
        prev = store.get_hash(doc.doc_id)
        if prev == h:
            report.skipped.append(doc.doc_id)
            continue

        # Ingest-time guardrail: flag (do not silently drop) docs that look like they carry instructions for the model.
        hits = scan_for_injection(doc.text)
        if hits:
            report.flagged.append(doc.doc_id)
            doc.metadata["injection_flags"] = ",".join(hits)

        chunks = chunk_document(doc, settings.chunk_target_tokens, settings.chunk_overlap_tokens)
        if settings.contextualize and llm is not None:
            from ragkit.ingest.contextualize import contextualize
            contextualize(doc, chunks, llm)
        vectors = embedder.embed([c.embed_text for c in chunks]) if chunks else []
        store.upsert_document(doc, h, chunks, vectors)
        report.chunks_embedded += len(chunks)
        (report.updated if prev else report.added).append(doc.doc_id)

    if full_reconcile:
        for gone in store.active_doc_ids(source) - seen:
            store.delete_document(gone)
            report.deleted.append(gone)
    return report
