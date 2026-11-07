# Configuration reference

`config.py` reads every setting from an environment variable that starts with `RAGKIT_`. The tables list each variable and its default.

## How settings load

- A variable in the shell wins over the same variable in `.env`.
- `ragkit` reads `.env` from the current folder only. Run it from the folder that holds the file.
- Quotes around a `.env` value are removed.
- Values are not validated. An unknown `RAGKIT_EMBED_PROVIDER` is treated as `openai`. An unknown `RAGKIT_RERANKER` is treated as `none`.

## LLM

| Variable | Default | What it does |
|---|---|---|
| `RAGKIT_LLM_BASE_URL` | `https://api.openai.com/v1` | Base URL of any OpenAI-compatible chat endpoint. |
| `RAGKIT_LLM_API_KEY` | empty | Sent as a bearer token. The LLM stays off while this is empty, and `ragkit` answers in retrieval-only mode. |
| `RAGKIT_LLM_MODEL` | `gpt-4o-mini` | Model name sent with each request. For Ollama it must match the pulled tag exactly (`llama3.1` is not `llama3.1:8b`). |

## Embeddings

| Variable | Default | What it does |
|---|---|---|
| `RAGKIT_EMBED_PROVIDER` | `hash` | `hash` is offline and not semantic, for tests only. `openai` uses any OpenAI-compatible `/embeddings` endpoint. `ollama` uses `http://localhost:11434/v1`. |
| `RAGKIT_EMBED_MODEL` | `text-embedding-3-small` | Model name for `openai` and `ollama`. |
| `RAGKIT_EMBED_BASE_URL` | `https://api.openai.com/v1` | Endpoint for `openai`. `ollama` ignores it. |
| `RAGKIT_EMBED_API_KEY` | empty | Bearer token for `openai`. `ollama` ignores it. |
| `RAGKIT_EMBED_DIM` | `384` | Vector size of the `hash` embedder. Other providers return their own size. |

## Chunking

| Variable | Default | What it does |
|---|---|---|
| `RAGKIT_CHUNK_TOKENS` | `350` | Target chunk size, counted as about 4 characters per token. |
| `RAGKIT_CHUNK_OVERLAP` | `50` | Tokens carried from the end of one chunk into the next. |
| `RAGKIT_CONTEXTUALIZE` | `false` | When `true`, an LLM call per chunk writes a short context line before embedding. It needs the LLM on and costs money. |

## Retrieval and reranking

| Variable | Default | What it does |
|---|---|---|
| `RAGKIT_CANDIDATES` | `30` | Results taken from each of BM25 and dense search before fusion. |
| `RAGKIT_RRF_K` | `60` | Smoothing constant in Reciprocal Rank Fusion. |
| `RAGKIT_TOP_K` | `6` | Chunks passed to the model after reranking. |
| `RAGKIT_RERANKER` | `none` | `none` or `cross-encoder`. The second needs the `[rerank]` extra. |
| `RAGKIT_RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder model name. |
| `RAGKIT_MIN_RELEVANCE` | `0.0` | The pipeline refuses when the best reranker score is below this value. It has no effect while `RAGKIT_RERANKER` is `none`. |

## Storage and audit

| Variable | Default | What it does |
|---|---|---|
| `RAGKIT_DB_PATH` | `./data/ragkit.db` | SQLite file that holds documents, chunks, ACLs, and vectors. |
| `RAGKIT_AUDIT_PATH` | `./data/audit.jsonl` | Append-only log with one JSON line per request. |

## API

| Variable | Default | What it does |
|---|---|---|
| `RAGKIT_INSECURE_HEADER_AUTH` | unset | `api.py` refuses to start unless this is `1`. It trusts the `x-user` and `x-groups` headers, so use it for local testing only. |
