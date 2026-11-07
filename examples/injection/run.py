"""Pattern 7 (retrieved text is untrusted): flag at ingest, screen at query time, keep the source fence intact.

needs: offline
Five public docs on one topic: an attack in a body paragraph, an attack in a heading, a body that tries to
close the <source> fence, an obfuscated instruction, and a clean doc. The demo checks what ingest flags, what
screening drops, that the fence cannot be broken, and prints the one gap a regex scanner leaves open.
Run: python examples/injection/run.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import Unavailable, build, cli, report  # noqa: E402

from ragkit import guardrails  # noqa: E402
from ragkit.audit import AuditLog  # noqa: E402
from ragkit.models import Principal  # noqa: E402
from ragkit.pipeline import RAGPipeline  # noqa: E402

DOCS = Path(__file__).parent / "docs"
QUERY = "travel expense receipts"
# heading_path is per chunk, so the attack heading governs the whole body here; a doc with a later clean section
# under another heading keeps that chunk.
ATTACKS = {"body-attack", "heading-attack"}


def main(embedder: str = "hash") -> bool:
    if embedder != "hash":
        raise Unavailable("injection is an offline demo: run it without --ollama")
    who = Principal("demo", frozenset())
    with tempfile.TemporaryDirectory() as tmp:
        settings, _, _, retriever, sync_report = build(Path(tmp), DOCS)
        settings.final_top_k = 10
        hits = retriever.retrieve([QUERY], who)
        kept, _ = guardrails.screen_context(hits)
        fenced = [guardrails.fence(i, h) for i, h in enumerate(kept, 1)]
        ans = RAGPipeline(retriever, settings, None, AuditLog(settings.audit_path)).ask(QUERY, who)

    retrieved = {h.chunk.doc_id for h in hits}
    kept_ids = {h.chunk.doc_id for h in kept}
    flagged = set(sync_report.flagged)
    ok = report(
        "ingest flags",
        f"sync flags exactly {sorted(ATTACKS)}",
        f"flagged = {sorted(flagged)}",
        flagged == ATTACKS,
    )
    ok &= report(
        "screening",
        "all 5 docs retrieved; screening drops body-attack and heading-attack, keeps clean and fence-break",
        f"retrieved = {sorted(retrieved)}; kept = {sorted(kept_ids)}",
        len(retrieved) == 5 and not kept_ids & ATTACKS and {"clean", "fence-break"} <= kept_ids,
    )
    counts = [(f.count("<source"), f.count("</source>")) for f in fenced]
    ok &= report(
        "fence",
        "every kept chunk is wrapped in exactly one <source ...> and one </source>, fence-break included",
        f"{len(fenced)} fenced chunks, (open, close) counts = {sorted(set(counts))}",
        bool(fenced) and set(counts) == {(1, 1)} and "fence-break" in kept_ids,
    )
    dropped = {e.split(":")[1] for e in ans.guardrail_events if e.startswith("context:")}
    ok &= report(
        "audit events",
        "the retrieval-only pipeline records a context:<doc_id>: event for each dropped doc",
        f"guardrail_events = {ans.guardrail_events}",
        dropped == ATTACKS,
    )
    miss = (
        "Its instruction is split with zero-width characters and spaces, so the regexes miss it. "
        if "obfuscated" not in flagged
        else "Its instruction is split with zero-width characters and spaces, but ingest flagged it anyway. "
    )
    print(
        f"KNOWN GAP (not asserted): obfuscated flagged at ingest = {'obfuscated' in flagged}; "
        f"would reach the model context = {'obfuscated' in kept_ids}. "
        + miss
        + "A regex scanner is only a first line; see docs/PATTERNS.md section 7 (add a classifier)."
    )
    return ok


if __name__ == "__main__":
    cli(main)
