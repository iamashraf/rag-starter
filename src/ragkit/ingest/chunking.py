"""Structure-aware chunking.

What the research and production write-ups agree on:
  * Simple recursive / heading-based splitting is a strong, cheap baseline.
    Fancier semantic chunking often does not beat it once you measure.
  * Chunks lose meaning when cut from their document ("the limit is $50" -- which limit?).
    Fix that cheaply with a breadcrumb (doc title > section > subsection) prepended to
    the text you embed, and optionally with LLM-written context (contextual retrieval).
  * Never split inside a code block or a table.

This chunker: split on markdown headings, then pack paragraphs up to a token budget
with overlap, keeping fenced code blocks and tables whole.
"""
from __future__ import annotations

import hashlib
import re

from ragkit.models import Chunk, SourceDocument

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def approx_tokens(text: str) -> int:
    # ~4 chars per token for English. Swap for tiktoken if you need precision.
    return max(1, len(text) // 4)


def _sections(text: str) -> list[tuple[list[str], str]]:
    """Return [(heading_path, body)] splitting on markdown headings, ignoring '#' inside code fences."""
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence = False

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append(([h for _, h in stack], body))
        buf.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def _blocks(body: str) -> list[str]:
    """Paragraph blocks, but code fences and tables stay as single atomic blocks."""
    blocks, cur, in_fence = [], [], False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            cur.append(line)
            if not in_fence:
                blocks.append("\n".join(cur))
                cur = []
            continue
        if in_fence:
            cur.append(line)
            continue
        if not line.strip():
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            # keep table rows glued together
            if cur and line.lstrip().startswith("|") and not cur[-1].lstrip().startswith("|"):
                blocks.append("\n".join(cur))
                cur = []
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return [b for b in blocks if b.strip()]


def chunk_document(doc: SourceDocument, target_tokens: int = 350, overlap_tokens: int = 50) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0
    for heading_path, body in _sections(doc.text) or [([], doc.text)]:
        breadcrumb = " > ".join([doc.title, *heading_path])
        window: list[str] = []
        size = 0

        def emit():
            nonlocal ordinal
            text = "\n\n".join(window).strip()
            if not text:
                return
            cid = hashlib.sha1(f"{doc.doc_id}:{ordinal}:{text}".encode()).hexdigest()[:20]
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc.doc_id,
                    ordinal=ordinal,
                    text=text,
                    embed_text=f"{breadcrumb}\n\n{text}",
                    heading_path=breadcrumb,
                    acl=list(doc.acl),
                    metadata={"source": doc.source, **{k: v for k, v in doc.metadata.items() if _scalar(v)}},
                )
            )
            ordinal += 1

        for block in _blocks(body):
            bt = approx_tokens(block)
            if window and size + bt > target_tokens:
                emit()
                # carry overlap from the tail of the previous window
                tail, tail_size = [], 0
                for prev in reversed(window):
                    if tail_size + approx_tokens(prev) > overlap_tokens:
                        break
                    tail.insert(0, prev)
                    tail_size += approx_tokens(prev)
                window, size = tail, tail_size
            window.append(block)
            size += bt
        emit()
    return chunks


def _scalar(v) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None
