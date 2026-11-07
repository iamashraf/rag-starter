"""Pattern 2 (hybrid retrieval): an exact identifier is matched lexically by BM25.

needs: offline
Six near-identical runbooks differ only by incident id. A query naming one id must put that runbook first.
Run: python examples/hybrid_identifier/run.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import build, cli, report  # noqa: E402

from ragkit.models import Principal  # noqa: E402

DOCS = Path(__file__).parent / "docs"
TARGETS = ["INC-7731", "INC-7733", "INC-7736"]


def dense_note(embedder: str) -> str:
    if embedder == "hash":
        return "not asserted; the hash embedder is not a semantic model"
    return "not asserted; informational, from the Ollama embedder (nomic-embed-text by default)"


def main(embedder: str = "hash") -> bool:
    ok = True
    dense_first = []
    principal = Principal("demo", frozenset())
    with tempfile.TemporaryDirectory() as tmp:
        _, store, emb, retriever, _ = build(Path(tmp), DOCS, embedder)
        for inc in TARGETS:
            q, target = f"{inc} what do I do", f"runbook-{inc.lower()}"
            bm25 = store.search_bm25(q, principal.acl_tokens, 10)
            top_bm25 = store.fetch([bm25[0][0]])[bm25[0][0]]["doc_id"] if bm25 else None
            exp = f"query {q!r}: top BM25 doc is {target}"
            seen = f"top BM25 doc is {top_bm25}" if bm25 else "the BM25 result was empty"
            ok &= report(f"bm25 {inc}", exp, seen, top_bm25 == target)
            top_fused = retriever.retrieve([q], principal)[0].chunk.doc_id
            exp = f"query {q!r}: top fused doc is {target}"
            ok &= report(f"fused {inc}", exp, f"top fused doc is {top_fused}", top_fused == target)
            dense = store.search_dense(emb.embed([q])[0], principal.acl_tokens, 100)
            ids = [store.fetch([c])[c]["doc_id"] for c, _ in dense]
            rank = ids.index(target) + 1 if target in ids else None
            note = dense_note(embedder)
            print(f"[dense {inc}] INFO ({note}): dense-only rank of {target} = {rank or 'absent'}")
            dense_first.append(rank == 1)
        if embedder == "hash":
            if all(dense_first):
                print(
                    "NOTE: on this corpus the dense side also ranked the target first, so this demo shows exact "
                    "ids being matched lexically, not dense search failing."
                )
            else:
                print(f"NOTE: dense side ranked the target first for {sum(dense_first)} of {len(dense_first)} ids.")
    return ok


if __name__ == "__main__":
    cli(main)
