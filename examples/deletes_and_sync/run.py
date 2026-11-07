"""Pattern 6 (freshness and deletes): sync skips unchanged docs, propagates deletes, and refuses a mass delete.

needs: offline
Runs five stages on one index built from a copy of the docs: re-sync is free, a removed file is deleted from the
index, an empty source raises SyncError (force=True overrides it), and an ACL edit re-indexes the doc.
Run: python examples/deletes_and_sync/run.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import Unavailable, build, cli, report  # noqa: E402

from ragkit.ingest.loaders import LocalFolderLoader  # noqa: E402
from ragkit.ingest.sync import SyncError, sync  # noqa: E402
from ragkit.models import Principal  # noqa: E402

DOCS = Path(__file__).parent / "docs"
FINANCE = Principal("fin", frozenset({"finance"}))
OPS = Principal("ops", frozenset({"ops"}))
SRE = Principal("sre", frozenset({"sre"}))


def main(embedder: str = "hash") -> bool:
    if embedder != "hash":
        raise Unavailable("deletes_and_sync is an offline demo: run it without --ollama")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "docs"
        shutil.copytree(DOCS, work)
        settings, store, emb, retriever, first = build(Path(tmp), work)

        def resync(**kw):
            return sync(LocalFolderLoader(work).load(), store, emb, settings, source="local", **kw)

        def ids(query, principal):
            return [h.chunk.doc_id for h in retriever.retrieve([query], principal)]

        again = resync()
        ok = report(
            "unchanged",
            "first sync adds 4 docs; a second sync skips all 4 and embeds 0 chunks",
            f"first: {first.summary()} | second: {again.summary()}",
            len(first.added) == 4 and len(again.skipped) == 4 and again.chunks_embedded == 0,
        )

        before = ids("quarterly budget forecast ledger", FINANCE)
        (work / "fin-budget.md").unlink()
        rep = resync()
        after = ids("quarterly budget forecast ledger", FINANCE)
        ok &= report(
            "delete",
            "deleted == ['fin-budget'] and the finance principal no longer retrieves it",
            f"deleted={rep.deleted}; fin-budget retrieved before: {'fin-budget' in before}, "
            f"after: {'fin-budget' in after}",
            rep.deleted == ["fin-budget"] and "fin-budget" in before and "fin-budget" not in after,
        )

        active = store.stats()["documents_active"]
        try:
            resync_empty = sync([], store, emb, settings, source="local")
            err = f"no error, {resync_empty.summary()}"
        except SyncError as e:
            err = f"SyncError: {e}"
        now = store.stats()["documents_active"]
        ok &= report(
            "empty source",
            f"sync([]) raises SyncError and active docs stay at {active}",
            f"{err}; active docs: {now}",
            err.startswith("SyncError") and now == active,
        )

        forced = sync([], store, emb, settings, source="local", force=True)
        left = store.stats()["documents_active"]
        ok &= report(
            "force",
            "force=True deletes all 3 remaining docs, leaving 0 active",
            f"deleted={sorted(forced.deleted)}; active docs: {left}",
            len(forced.deleted) == 3 and left == 0,
        )

        shutil.copytree(DOCS, work, dirs_exist_ok=True)
        back = resync()
        runbook = work / "ops-runbook.md"
        runbook.write_text(runbook.read_text().replace("group:ops", "group:sre"))
        rep = resync()
        q = "failover pager rotation"
        new_ids, old_ids = ids(q, SRE), ids(q, OPS)
        ok &= report(
            "acl change",
            "4 docs restored as added; the acl edit updates only ops-runbook; group:sre reads it, group:ops does not",
            f"restored added={len(back.added)}; updated={rep.updated}; sre sees {new_ids}; "
            f"old ops group sees {old_ids}",
            len(back.added) == 4 and rep.updated == ["ops-runbook"]
            and "ops-runbook" in new_ids and "ops-runbook" not in old_ids,
        )
    return ok


if __name__ == "__main__":
    cli(main)
