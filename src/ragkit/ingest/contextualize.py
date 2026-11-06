"""Contextual retrieval (optional, ingest-time).

Idea: ask an LLM to write 1-2 sentences that situate each chunk inside its whole
document, and prepend that to the text you embed and BM25-index. Anthropic reported
this, combined with BM25 and reranking, cut top-20 retrieval failures by about two thirds.

Cost: one LLM call per chunk. Use prompt caching on the document part if your
provider supports it, and only re-run for chunks whose content hash changed.
"""
from __future__ import annotations

from ragkit.llm import LLMClient
from ragkit.models import Chunk, SourceDocument

PROMPT = """<document>
{doc}
</document>
Here is a chunk from that document:
<chunk>
{chunk}
</chunk>
Write one or two short sentences that situate this chunk within the overall document,
to improve search retrieval of the chunk. Name the specific system, policy, team or
product it is about. Answer with only the context sentences."""


def contextualize(doc: SourceDocument, chunks: list[Chunk], llm: LLMClient, max_doc_chars: int = 24000) -> None:
    doc_text = doc.text[:max_doc_chars]
    for c in chunks:
        ctx = llm.complete(
            [{"role": "user", "content": PROMPT.format(doc=doc_text, chunk=c.text)}],
            max_tokens=120,
            temperature=0,
        ).strip()
        if ctx:
            c.embed_text = f"{c.heading_path}\n{ctx}\n\n{c.text}"
            c.metadata["context"] = ctx
