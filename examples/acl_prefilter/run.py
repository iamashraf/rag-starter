"""Pattern 5 (permissions in the query): the ACL filter runs inside search, not on the top-k afterwards.

needs: offline
Twenty restricted HR docs repeat the query terms and fill the top 3. One public doc matches less strongly.
A post-filter on the top 3 returns nothing for an employee; filtering inside the query still finds the public doc.
Run: python examples/acl_prefilter/run.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import Unavailable, build, cli, report  # noqa: E402

from ragkit.models import Principal, RetrievedChunk  # noqa: E402

DOCS = Path(__file__).parent / "docs"
QUERY = "annual leave carryover"
AUDITOR = Principal("auditor", frozenset({"hr-leadership"}))


def post_filter(retriever, query: str, principal: Principal, auditor: Principal) -> list[RetrievedChunk]:
    """The anti-pattern: retrieve the top-k as someone who sees everything, then drop what `principal` may not read."""
    hits = retriever.retrieve([query], auditor)
    return [h for h in hits if set(h.chunk.acl) & principal.acl_tokens]


def main(embedder: str = "hash") -> bool:
    if embedder != "hash":
        raise Unavailable("acl_prefilter is an offline demo: run it without --ollama")
    emp = Principal("emp", frozenset())
    print("NOTE: ragkit has no post-filter path. post_filter() below is this demo's own illustration of the anti-pattern.")
    with tempfile.TemporaryDirectory() as tmp:
        settings, _, _, retriever, _ = build(Path(tmp), DOCS)
        settings.final_top_k = 3
        post = post_filter(retriever, QUERY, emp, AUDITOR)
        inq = retriever.retrieve([QUERY], emp)
        top_audit = [h.chunk.doc_id for h in retriever.retrieve([QUERY], AUDITOR)]
    post_ids, inq_ids = [h.chunk.doc_id for h in post], [h.chunk.doc_id for h in inq]
    ok = report(
        "post-filter",
        f"query {QUERY!r} as emp: top 3 are all restricted, so post-filtering leaves 0 chunks",
        f"auditor top 3 = {top_audit}; after filtering for emp: {len(post)} chunks {post_ids}",
        len(post) == 0,
    )
    ok &= report(
        "in-query filter",
        "filtering inside the search returns a chunk from pol-leave",
        f"{len(inq)} chunks {inq_ids}",
        "pol-leave" in inq_ids,
    )
    leaked = [i for i in post_ids + inq_ids if i.startswith("hr-restricted")]
    ok &= report("no leak", "no restricted doc in either result for emp", f"restricted ids found: {leaked}", not leaked)
    return ok


if __name__ == "__main__":
    cli(main)
