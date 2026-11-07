"""Pattern 2 (hybrid retrieval): dense vectors find a paraphrase that BM25 cannot.

needs: Ollama with the nomic-embed-text model (python examples/hybrid_paraphrase/run.py)
Each question is a short phrase of content words that share no stemmed term with the target doc, so BM25 cannot
rank it in its top 3. The dense embedder must rank it first, and the fused result must contain it.
ragkit sends no query/document prefixes to nomic-embed-text. Results are measured for this embedder and
this corpus only; they are not a general claim about dense retrieval.
See DISCLOSURE below: the corpus is hand-built, so this is a best case, not a benchmark.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import Unavailable, build, cli, report  # noqa: E402

from ragkit.models import Principal  # noqa: E402

HERE = Path(__file__).parent
DISCLOSURE = (
    "NOTE: the docs and questions were written by us so that no question shares a word with its target doc. "
    "That makes this a constructed best case for dense search. "
    "6 questions over 12 docs is not a benchmark; results hold for nomic-embed-text and this corpus only."
)


def rank_of(store, hits, target):
    ids = [store.fetch([c])[c]["doc_id"] for c, _ in hits]
    return ids, (ids.index(target) + 1 if target in ids else None)


def main(embedder: str = "ollama") -> bool:
    if embedder != "ollama":
        raise Unavailable("hybrid_paraphrase needs --ollama: the hash embedder cannot match a paraphrase")
    print(DISCLOSURE)
    ok = True
    principal = Principal("demo", frozenset())
    golden = [json.loads(line) for line in (HERE / "golden.jsonl").read_text().splitlines() if line.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        _, store, emb, retriever, _ = build(Path(tmp), HERE / "docs", embedder)
        for g in golden:
            q, target = g["question"], g["expected_docs"][0]
            ids, rank = rank_of(store, store.search_bm25(q, principal.acl_tokens, 50), target)
            ok &= report(f"bm25 {g['id']}", f"query {q!r}: {target} is not in the BM25 top 3",
                         f"BM25 top 3 {ids[:3]}, rank of {target}: {rank or 'absent'}", target not in ids[:3])
            hits = store.search_dense(emb.embed([q])[0], principal.acl_tokens, 50)
            ids, rank = rank_of(store, hits, target)
            ok &= report(f"dense {g['id']}", f"query {q!r}: dense-only rank of {target} is 1",
                         f"dense-only rank of {target}: {rank}", rank == 1)
            fused = [r.chunk.doc_id for r in retriever.retrieve([q], principal)]
            rank = fused.index(target) + 1 if target in fused else None
            ok &= report(f"fused {g['id']}", f"query {q!r}: fused results contain {target}",
                         f"fused doc order {fused}, rank of {target}: {rank}", rank is not None)
    return ok


if __name__ == "__main__":
    cli(main, default="ollama")
