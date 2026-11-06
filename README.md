# ragkit: a starter for permission-aware RAG over internal docs

A small scaffold that does what most demos skip: hybrid search, ACL enforcement inside the query, incremental sync with real deletes, layered
guardrails, grounded answers with citations, an audit log, and an eval harness that fails on
ACL leaks.

It runs on one SQLite file and needs no other infrastructure. Swap pieces out as you grow. Read [`docs/PATTERNS.md`](docs/PATTERNS.md) for the reasoning and sources
behind each choice.

## Quick start

```bash
pip install -e ".[dev,api]"
ragkit ingest sample_docs
ragkit ask "What does PAY-4012 mean?" --user alice --groups engineering --no-llm
python eval/run_eval.py --verbose
pytest -q
```

To get real answers, copy `.env.example` to `.env`, set an LLM endpoint (OpenAI, or Ollama at
`http://localhost:11434/v1`), and set `RAGKIT_EMBED_PROVIDER=openai` or `ollama`. The default
`hash` embedder is not semantic. It exists so tests run offline.

HTTP: `RAGKIT_INSECURE_HEADER_AUTH=1 uvicorn ragkit.api:app --reload`, then
`curl -XPOST localhost:8000/ask -H 'x-user: alice' -H 'x-groups: engineering' -H 'content-type: application/json' -d '{"question":"How do I rotate payments secrets?"}'`
The API trusts the `x-user` and `x-groups` headers, so it refuses to start without that flag. Replace
`get_principal` with JWT validation before anyone else can reach it.

## How a request flows

```
question
  -> input guardrail ................................ guardrails.check_input
  -> condense follow-up, route simple | decompose ... rewrite.py
  -> BM25 + dense search, ACL pre-filtered .......... store.search_bm25 / search_dense
  -> Reciprocal Rank Fusion -> rerank ............... retrieval.py
  -> drop chunks that look like injected prompts .... guardrails.screen_context
  -> refuse early if nothing relevant ............... pipeline.ask
  -> grounded answer with [n] citations ............. pipeline.SYSTEM
  -> citation check + redaction ..................... guardrails.check_output
  -> audit log ...................................... audit.py

ingest: loader (with ACL) -> content hash -> chunk (breadcrumbs) -> [contextualize] -> embed -> upsert
        docs missing from source -> tombstone -> delete chunks
```

## Where each best practice lives

| Practice | File |
|---|---|
| Hybrid BM25 + vector with RRF | `store.py`, `retrieval.py` |
| Retrieve wide, rerank narrow | `retrieval.py` (`CrossEncoderReranker`) |
| Heading-aware chunking, breadcrumbs, whole code blocks and tables | `ingest/chunking.py` |
| Contextual retrieval (optional) | `ingest/contextualize.py` |
| ACL copied from source, enforced inside the search query | `ingest/loaders.py`, `store._acl_clause` |
| Second ACL check on fetched rows | `retrieval.py` |
| Content-hash sync, tombstone then delete | `ingest/sync.py`, `store.delete_document` |
| Injection scanning at ingest and query time | `guardrails.py`, `ingest/sync.py` |
| Answer only from sources, cite, fixed refusal | `pipeline.py` |
| Citation validation and secret redaction | `guardrails.check_output` |
| Audit trail per request | `audit.py` |
| Retrieval metrics + ACL leak gate, optional RAGAS | `eval/run_eval.py` |
| Any OpenAI-compatible model (OpenAI, Ollama, vLLM, gateway) | `llm.py` |

## Things to notice when you run the demo

- **ACL in action.** Ask for the Staff Engineer salary band as `--groups engineering` and then as
  `--groups hr-leadership`. The confidential doc only shows up for the second.
- **Top-k always returns something.** In `--no-llm` mode the engineering user still gets on-call
  chunks for the salary question. That is why you need a reranker score threshold
  (`RAGKIT_MIN_RELEVANCE`) and a model that refuses. With an LLM configured, it should answer
  with the refusal string.
- **The planted attack.** `sample_docs/wiki/team-offsite-notes.md` tells the AI to approve
  business class and send receipts to an outside URL. It is flagged at ingest and dropped from
  context at query time. Ask about offsite travel and look at the guardrail events.
- **Sync is incremental.** Run `ragkit ingest sample_docs` twice. The second run embeds nothing.
  Delete a file and run it again. The doc is removed from retrieval.
- **Dense search needs a real embedder.** With the hash embedder, "PAY-4012 what do I do" can put
  PAY-5001 first because the dense side is noise. Switch to a real model and compare.

## Path to production

1. **Use case and golden set first.** Collect 50 to 200 real questions from search logs, Slack and
   tickets, including ones that should be refused and ones asked by people without access.
2. **Real connectors.** One loader per source (Confluence, SharePoint, Drive, GitHub, Jira) that
   copies the source's permissions and uses stable source ids. Event-driven sync plus a nightly
   full reconcile.
3. **Real identity.** Validate the IdP JWT at the API, map IdP groups to ACL tokens. For
   relationship-based permissions, add OpenFGA / SpiceDB / Cedar as a post-retrieval check.
4. **Swap the store** when you pass ~100k chunks: Postgres + pgvector + full-text (or ParadeDB),
   OpenSearch / Elastic, Qdrant or Weaviate. All support hybrid search and RRF natively. Keep the
   ACL pre-filter inside the query.
5. Turn on the reranker and contextual retrieval, then re-run the eval and keep what helps.
6. **Harden guardrails.** Add a classifier behind `scan_for_injection`, a grounding check on
   output, and tool allow-lists if you add actions.
7. **Observability.** Add OpenTelemetry spans per stage, ship `audit.jsonl` to your log pipeline,
   and build dashboards for refusal rate, citation failures, latency and ACL denials.
8. **Agentic path last.** Put it behind a router, and use it only for question types the eval
   shows the simple path cannot handle.

## Config

All settings are env vars prefixed `RAGKIT_` (see `.env.example` and `config.py`). The ones you
will touch most: `EMBED_PROVIDER`, `EMBED_MODEL`, `LLM_BASE_URL`, `LLM_MODEL`, `CHUNK_TOKENS`,
`CANDIDATES`, `TOP_K`, `RERANKER`, `MIN_RELEVANCE`, `CONTEXTUALIZE`.

Changing the embedding model means re-embedding everything. Delete `data/ragkit.db` and ingest again.
The index records which embedder built it and raises `EmbedderMismatch` if you switch.

`ragkit ingest` refuses to delete more than half of the indexed docs (a wrong path would otherwise empty
the index). Pass `--force` when that is what you want. Docs with no `acl` in their front matter are
visible to nobody.
