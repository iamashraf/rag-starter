# ragkit: a permission-aware RAG starter for internal documents

![CI](https://github.com/iamashraf/rag-starter/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/github/license/iamashraf/rag-starter)

ragkit is a Python starter kit for retrieval-augmented generation (RAG) over internal documents. It enforces who can read what inside the search query, combines BM25 and vector search, blocks prompt injection from retrieved text, and includes an eval that fails on any permission leak. It runs on one SQLite file and works with any OpenAI-compatible model, including Ollama.

## What it does

You ask a question as a user with a set of groups. ragkit searches only the documents those groups may read, then answers from them with `[n]` citations. If the documents do not contain the answer, it replies with a fixed refusal string.

## Why this exists

Most RAG demos skip the parts that fail a security review: permissions, deletes, and injected instructions in documents. ragkit puts those parts in a small codebase you can read, then lets you swap pieces out as you grow. [`docs/PATTERNS.md`](docs/PATTERNS.md) gives the reasoning and sources behind each choice.

## Features

- Hybrid search: BM25 and vectors, fused with Reciprocal Rank Fusion, with an optional cross-encoder reranker.
- Permission filtering inside the search query, plus a second check on the fetched rows.
- Heading-aware chunking that keeps code blocks and tables whole, with optional LLM-written context per chunk.
- Incremental sync by content hash. Deleted documents are removed from search, and a sync that would delete over half the index asks for `--force`.
- Prompt-injection scanning at ingest and at query time.
- Answers only from the sources, with citation checks, secret redaction, and a fixed refusal string.
- An audit log with one JSON line per request.
- An eval that reports hit@k, recall@k, and MRR, and fails on any permission leak. RAGAS scoring is optional.

## Quick start

You need Python 3.11 or newer.

```bash
pip install -e ".[dev,api]"
ragkit ingest sample_docs
ragkit ask "What does PAY-4012 mean?" --user alice --groups engineering --no-llm
python eval/run_eval.py --verbose
pytest -q
```

This runs offline with the `hash` embedder, which is not semantic. For real answers, see [Configuration](#configuration).

## Usage

Ask from the command line:

```bash
ragkit ask "What are the on-call severity levels?" --user alice --groups engineering --no-llm
```

Or start the HTTP API and call it:

```bash
RAGKIT_INSECURE_HEADER_AUTH=1 uvicorn ragkit.api:app --reload
curl -XPOST localhost:8000/ask -H 'x-user: alice' -H 'x-groups: engineering' -H 'content-type: application/json' -d '{"question":"How do I rotate payments secrets?"}'
```

The API trusts the `x-user` and `x-groups` headers, so it refuses to start without that flag. Replace
`get_principal` with JWT validation before anyone else can reach it.

### Things to notice in the demo

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
  Delete a file and run it again. The doc is removed from retrieval. If a run would delete more than
  half of the indexed docs, it stops and asks for `--force`.
- **Dense search needs a real embedder.** With the hash embedder, "PAY-4012 what do I do" can put
  PAY-5001 first because the dense side is noise. Switch to a real model and compare.

## Architecture

### How a request flows

```
question
  -> input guardrail ................................ guardrails.check_input
  -> condense follow-up, route simple | decompose ... rewrite.py
  -> BM25 + dense search, ACL pre-filtered .......... store.search_bm25 / search_dense
  -> Reciprocal Rank Fusion -> rerank ............... retrieval.py
  -> drop chunks that look like injected prompts .... guardrails.screen_context
  -> refuse early if nothing relevant (reranker on) .... pipeline.ask
  -> grounded answer with [n] citations ............. pipeline.SYSTEM
  -> citation check + redaction ..................... guardrails.check_output
  -> audit log ...................................... audit.py

ingest: loader (with ACL) -> content hash -> chunk (breadcrumbs) -> [contextualize] -> embed -> upsert
        docs missing from source -> tombstone -> delete chunks
```

### Where each feature lives

| Feature | File |
|---|---|
| Hybrid BM25 + vector with RRF | `store.py`, `retrieval.py` |
| Retrieve wide, rerank narrow | `retrieval.py` (`CrossEncoderReranker`) |
| Heading-aware chunking, breadcrumbs, whole code blocks and tables | `ingest/chunking.py` |
| Contextual retrieval (optional) | `ingest/contextualize.py` |
| ACL copied from source, enforced inside the search query | `ingest/loaders.py`, `store._acl_clause` |
| Second ACL check on fetched rows | `retrieval.py` |
| Content-hash sync, tombstone then delete | `ingest/sync.py`, `store.delete_document` |
| Mass-delete guard, index tied to its embedder | `ingest/sync.py`, `store.bind_embedder` |
| Injection scanning at ingest and query time | `guardrails.py`, `ingest/sync.py` |
| Answer only from sources, cite, fixed refusal | `pipeline.py` |
| Citation validation and secret redaction | `guardrails.check_output` |
| Audit trail per request | `audit.py` |
| Retrieval metrics + ACL leak gate, optional RAGAS | `eval/run_eval.py` |
| Any OpenAI-compatible model (OpenAI, Ollama, vLLM, gateway) | `llm.py` |

## Configuration

Every setting is an environment variable that starts with `RAGKIT_`. Put them in a `.env` file in the folder where you run `ragkit`, or export them in your shell. [`docs/CONFIG.md`](docs/CONFIG.md) lists every variable with its default.

### Pick a setup

The default needs no `.env`. It uses the offline `hash` embedder, so run `ragkit ask --no-llm` to see retrieval alone.

For OpenAI, set your own keys. Keep them in `.env`, which git ignores, and never commit them.

```bash
RAGKIT_LLM_API_KEY=
RAGKIT_EMBED_PROVIDER=openai
RAGKIT_EMBED_API_KEY=
```

For Ollama, set:

```bash
RAGKIT_LLM_BASE_URL=http://localhost:11434/v1
RAGKIT_LLM_API_KEY=ollama
RAGKIT_LLM_MODEL=llama3.1
RAGKIT_EMBED_PROVIDER=ollama
RAGKIT_EMBED_MODEL=nomic-embed-text
```

The LLM turns on when `RAGKIT_LLM_API_KEY` is not empty. Ollama ignores the key, but it must be set.

### Before you change a setting

- A new embedding model needs a new index. `ragkit` raises `EmbedderMismatch` if you switch. Delete `data/ragkit.db` and run `ragkit ingest` again.
- `ragkit ingest` refuses to delete more than half of the indexed docs, because a wrong path would empty the index. Pass `--force` when you mean it.
- A doc with no `acl` in its front matter is visible to nobody.
- `uvicorn ragkit.api:app` needs `RAGKIT_INSECURE_HEADER_AUTH=1` until you replace `get_principal` with JWT validation.

## Roadmap

ragkit does not include the items below yet. They are the steps from this starter to a production system.

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

## License

MIT. See [`LICENSE`](LICENSE).
