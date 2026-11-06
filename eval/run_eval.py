"""Evaluation harness. Run it on every change to chunking, embeddings, retrieval or prompts.

Two layers, measured separately because they fail separately:

1. Retrieval (no LLM needed, cheap, deterministic):
   hit@k, recall@k, MRR against expected source docs, plus ACL leak count
   (any retrieval of a must_not_retrieve doc is a hard failure, gate CI on it).

2. Generation (needs an LLM): pass --ragas to score faithfulness, answer relevancy,
   context precision and context recall with RAGAS. Treat LLM-judge scores as
   trend lines, not absolute truth; spot-check them by hand.

Build the golden set from real employee questions (search logs, Slack, support tickets),
not from questions you invent while looking at the docs.

    python eval/run_eval.py --golden eval/golden.jsonl
    python eval/run_eval.py --golden eval/golden.jsonl --ragas
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragkit.app import build_app
from ragkit.models import Principal


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def retrieval_metrics(golden: list[dict], k: int) -> dict:
    app = build_app(use_llm=False)
    app.settings.final_top_k = k
    hits = recall_sum = mrr_sum = 0.0
    scored = leaks = 0
    rows = []
    for g in golden:
        principal = Principal(f"eval-{g['id']}", frozenset(g.get("groups", [])))
        items = app.pipeline.retriever.retrieve([g["question"]], principal)
        docs = [it.chunk.doc_id for it in items]
        uniq = list(dict.fromkeys(docs))
        leaked = sorted(set(g.get("must_not_retrieve", [])) & set(docs))
        leaks += bool(leaked)
        exp = set(g.get("expected_docs", []))
        if exp:
            scored += 1
            hit = bool(exp & set(uniq))
            hits += hit
            recall_sum += len(exp & set(uniq)) / len(exp)
            rr = next((1 / (i + 1) for i, d in enumerate(uniq) if d in exp), 0.0)
            mrr_sum += rr
        rows.append({"id": g["id"], "retrieved_docs": uniq, "leaked": leaked})
    n = max(scored, 1)
    return {
        f"hit@{k}": round(hits / n, 3),
        f"recall@{k}": round(recall_sum / n, 3),
        "mrr": round(mrr_sum / n, 3),
        "acl_leaks": leaks,
        "questions": len(golden),
        "rows": rows,
    }


def ragas_eval(golden: list[dict]) -> dict:  # pragma: no cover - needs network + ragas
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    app = build_app()
    if app.llm is None:
        sys.exit("Set RAGKIT_LLM_API_KEY to run generation evals")
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for g in golden:
        if not g.get("expected_docs"):
            continue
        ans = app.pipeline.ask(g["question"], Principal("eval", frozenset(g.get("groups", []))))
        data["question"].append(g["question"])
        data["answer"].append(ans.text)
        data["contexts"].append([c.chunk.text for c in ans.citations] or [""])
        data["ground_truth"].append(g["reference"])
    result = evaluate(Dataset.from_dict(data), metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    return {k: round(float(v), 3) for k, v in result.items()} if hasattr(result, "items") else {"result": str(result)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default="eval/golden.jsonl")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--ragas", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    golden = load(args.golden)
    m = retrieval_metrics(golden, args.k)
    rows = m.pop("rows")
    print(json.dumps(m, indent=2))
    if args.verbose:
        for r in rows:
            print(r)
    if args.ragas:
        print(json.dumps(ragas_eval(golden), indent=2))
    if m["acl_leaks"]:
        sys.exit("FAIL: ACL leak detected")


if __name__ == "__main__":
    main()
