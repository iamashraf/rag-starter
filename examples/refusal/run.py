"""Pattern 8 (grounded generation with an explicit refusal): ragkit turns a refusal into Answer.refused.

needs: offline for part A; ollama optional for part B (--ollama)
Part A uses ScriptedLLM, a fake that returns a fixed string, so it tests how ragkit handles a model's output, not
how a model behaves. Part B asks a real local model the same unanswerable question and only prints what happened.
Run: python examples/refusal/run.py [--ollama]
"""
import os
import sys
import tempfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import ScriptedLLM, build, cli, make_settings, report, require_ollama  # noqa: E402

from ragkit.audit import AuditLog  # noqa: E402
from ragkit.guardrails import REFUSAL  # noqa: E402
from ragkit.ingest.loaders import LocalFolderLoader  # noqa: E402
from ragkit.ingest.sync import sync  # noqa: E402
from ragkit.llm import build_embedder, build_llm  # noqa: E402
from ragkit.models import Principal  # noqa: E402
from ragkit.pipeline import RAGPipeline  # noqa: E402
from ragkit.retrieval import HybridRetriever  # noqa: E402
from ragkit.store import Store  # noqa: E402

DOCS = Path(__file__).parent / "docs"
USER = Principal("emp", frozenset())
QUESTION = "What is the favorite color of the CEO?"  # the expense docs never say


def ollama_version() -> str:
    try:
        return httpx.get("http://localhost:11434/api/version", timeout=5).json()["version"]
    except (httpx.HTTPError, ValueError, KeyError):
        return "unknown"


def main(embedder: str = "hash") -> bool:
    print("PART A (offline: hash embedder, ScriptedLLM, runs the same with or without --ollama)")
    print("Note: ScriptedLLM is a fake that returns a fixed string. Checks 1, 2 and 4 test how ragkit handles a")
    print("model's output, not how a model behaves.")
    with tempfile.TemporaryDirectory() as tmp:
        settings, _, _, retriever, _ = build(Path(tmp), DOCS)

        def ask(llm, question=QUESTION):
            return RAGPipeline(retriever, settings, llm, AuditLog(settings.audit_path)).ask(question, USER)

        ans = ask(ScriptedLLM(REFUSAL))
        ok = report(
            "refusal text",
            "the exact refusal string gives refused=True and no citations",
            f"refused={ans.refused}; citations={len(ans.citations)}",
            ans.refused is True and ans.citations == [],
        )

        ans = ask(ScriptedLLM(REFUSAL.rstrip(".") + " but [1] says 90 days"))
        ok &= report(
            "refusal quoted",
            "a longer answer that quotes the refusal is still an answer: refused=False, citations kept",
            f"refused={ans.refused}; citations={len(ans.citations)}",
            ans.refused is False and len(ans.citations) > 0,
        )

        ans = ask(None)
        ok &= report(
            "no llm",
            "retrieval-only mode never refuses: top-k always returns something, so refused=False with citations. "
            "This is why the project needs a model that refuses or a reranker threshold",
            f"refused={ans.refused}; citations={len(ans.citations)}",
            ans.refused is False and len(ans.citations) > 0,
        )

        ans = ask(ScriptedLLM("The limit is 5 days [9]"))
        ok &= report(
            "bad citation",
            "citing source [9] when fewer exist adds an output:invalid_citations guardrail event",
            f"events={ans.guardrail_events}",
            any(e.startswith("output:invalid_citations") for e in ans.guardrail_events),
        )

    if embedder == "ollama":
        observe_ollama()
    return ok


def observe_ollama() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        s = make_settings(Path(tmp), "ollama")
        s.llm_base_url = os.environ.get("RAGKIT_LLM_BASE_URL", "http://localhost:11434/v1")
        s.llm_api_key = "ollama"
        s.llm_model = os.environ.get("RAGKIT_LLM_MODEL", "llama3.1:8b")
        require_ollama([s.llm_model])
        store, emb = Store(s.db_path), build_embedder(s)
        sync(LocalFolderLoader(DOCS).load(), store, emb, s, source="local")
        pipe = RAGPipeline(HybridRetriever(store, emb, s), s, build_llm(s), AuditLog(s.audit_path))
        ans = pipe.ask(QUESTION, USER)
    print("\nPART B: observation of one real model run, not a guarantee. Another run or model may answer differently.")
    print(f"question: {QUESTION}")
    print(f"refused: {ans.refused}")
    print(f"answer: {ans.text}")
    print(f"llm model: {s.llm_model} | embedding model: {s.embed_model} | ollama version: {ollama_version()}")


if __name__ == "__main__":
    cli(main)
