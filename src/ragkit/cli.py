"""CLI:
    ragkit ingest ./sample_docs
    ragkit ask "How do I rotate the payments-api secrets?" --user alice --groups engineering
    ragkit stats
"""
from __future__ import annotations

import argparse
import json

from ragkit.app import build_app
from ragkit.ingest.loaders import LocalFolderLoader
from ragkit.ingest.sync import sync
from ragkit.models import Principal


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ragkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="sync a local folder into the index")
    pi.add_argument("folder")
    pi.add_argument("--no-reconcile", action="store_true", help="do not delete docs missing from the folder")

    pa = sub.add_parser("ask")
    pa.add_argument("question")
    pa.add_argument("--user", default="demo")
    pa.add_argument("--groups", default="", help="comma separated, e.g. engineering,oncall")
    pa.add_argument("--no-llm", action="store_true", help="retrieval only")
    pa.add_argument("--json", action="store_true")

    sub.add_parser("stats")
    args = p.parse_args(argv)

    if args.cmd == "ingest":
        app = build_app()
        report = sync(LocalFolderLoader(args.folder).load(), app.store, app.embedder, app.settings,
                      source="local", full_reconcile=not args.no_reconcile, llm=app.llm)
        print(report.summary())
        if report.flagged:
            print("flagged for review (possible prompt injection):", ", ".join(report.flagged))
    elif args.cmd == "ask":
        app = build_app(use_llm=not args.no_llm)
        principal = Principal(args.user, frozenset(g for g in args.groups.split(",") if g))
        ans = app.pipeline.ask(args.question, principal)
        if args.json:
            print(json.dumps({"answer": ans.text, "refused": ans.refused, "events": ans.guardrail_events,
                              "sources": [{"title": c.title, "uri": c.uri, "section": c.chunk.heading_path,
                                           "signals": c.signals} for c in ans.citations]}, indent=2))
        else:
            print(ans.text)
            if ans.citations and app.llm:
                print("\nSources:")
                for i, c in enumerate(ans.citations, 1):
                    print(f"  [{i}] {c.chunk.heading_path}  ({c.uri})")
            if ans.guardrail_events:
                print("\nguardrail events:", ans.guardrail_events)
    elif args.cmd == "stats":
        print(json.dumps(build_app(use_llm=False).store.stats(), indent=2))


if __name__ == "__main__":
    main()
