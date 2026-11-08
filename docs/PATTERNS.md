# RAG for internal company knowledge: patterns and lessons

Research notes collected November 2026 from production write-ups, vendor docs and recent papers.
Each section says what the pattern is, why it matters for internal knowledge specifically,
where it lives in this scaffold, and what to do next.

---

## 1. Start from the use case, not the vector database

The recurring advice: a demo is easy, a reliable internal assistant is hard, and the use case
decides the architecture. "Answer HR policy questions for 3,000 employees from Confluence,
limited by department" tells you the sources, the ACL model, the freshness need and the
eval set. "Which vector DB?" tells you nothing.

**In this repo:** nothing to configure. Write your one-paragraph use case at the top of your
fork's README before you change any code.

---

## 2. Hybrid retrieval (BM25 + dense) is the baseline

Internal queries are full of exact tokens: ticket ids, service names, error codes, acronyms,
policy numbers. Embeddings blur those. BM25 matches them exactly but misses paraphrase.
Most real queries need both.

- One write-up reports accuracy rising from about 60% to 85% on a 50-question set after adding
  BM25 next to vectors. Public benchmarks show gains anywhere from a couple of
  percent to 20%+ depending on the corpus. Measure on yours.
- Reciprocal Rank Fusion (RRF) is now the built-in fusion in Elastic, OpenSearch, Weaviate,
  Qdrant and Azure AI Search. It uses rank positions only, so no score normalization.
- Learned sparse models (SPLADE, BGE-M3 sparse) are an option, but for corpora heavy with
  ids and numbers, a real BM25 index is still the safe choice.

**In this repo:** `store.py` (FTS5 BM25 + vectors), `retrieval.py` (`rrf`, `HybridRetriever`).
Try `ragkit ask "PAY-4012" --no-llm` and look at `bm25_rank` in `--json` output.

**Demos:** `examples/hybrid_identifier` (BM25 finds an exact id) and `examples/hybrid_paraphrase` (dense finds a paraphrase, needs Ollama).

---

## 3. Retrieve wide, rerank narrow

Bi-encoder embeddings compress a chunk into a single vector. A cross-encoder reranker reads the
query and chunk together and is much more precise, but too slow to run on the whole corpus.
Standard funnel: ~30 BM25 + ~30 dense candidates, fuse, rerank, keep 5 to 8.

Anthropic's contextual retrieval study reports that contextual embeddings plus BM25 cut top-20
retrieval failures by 49%, and adding reranking cut them by 67%.

A reranker cannot rescue a chunk that stage one never found, so keep stage one broad.

**In this repo:** `retrieval.py` (`CrossEncoderReranker`), enable with `RAGKIT_RERANKER=cross-encoder`.
Good open models: `BAAI/bge-reranker-v2-m3`, `mixedbread-ai/mxbai-rerank-base`. Hosted: Cohere, Voyage.

---

## 4. Chunking: simple and structure-aware beats clever

- Recursive / heading-based splitting is a strong, cheap baseline. A 2026 study on long
  structured documents found cluster-based semantic chunking did not beat simpler strategies.
  Other experiments show semantic chunking helping on some content. Measure before you switch.
- The real problem is lost context. "The limit is 75 USD" means nothing without "Expense policy >
  Meal limits". Two fixes:
  - **Breadcrumbs.** Prepend `Doc title > Section > Subsection` to the text you embed. This
    costs nothing.
  - **Contextual retrieval**: an LLM writes 1 to 2 sentences placing each chunk in its document,
    prepended before embedding and BM25 indexing. It improves retrieval but costs one LLM call per
    chunk. Late chunking (embed the whole doc, then pool per chunk) is cheaper but scored lower
    in tests.
- Never split inside code blocks or tables. Runbooks are full of both. The one exception is a block
  more than four times the chunk size, which an embedding call would reject. The chunker splits it
  on lines.

**In this repo:** `ingest/chunking.py` (heading-aware, breadcrumb, atomic code/tables),
`ingest/contextualize.py` (turn on with `RAGKIT_CONTEXTUALIZE=true`).

---

## 5. Permissions must be enforced at retrieval time, inside the query

This is the item that fails enterprise security reviews. If the pipeline does not enforce the
source system's permissions, a user can retrieve a document they cannot open, and the LLM will
summarize it confidently with no sign anything went wrong.

Practices to follow:

- **Pre-filter, not post-filter.** Push the ACL into the search query. Filtering the top-k
  afterwards both risks leaks (if anyone forgets the filter) and starves results.
- **Copy ACLs from the source system** at ingest (Confluence space + page restrictions, Drive
  sharing, SharePoint groups, GitHub team access). Prefer inheritance; use per-document
  exceptions only where the source has them.
- **Real ACLs are graphs.** Static role tags break when access depends on project membership or
  ad hoc sharing. For that, put a fine-grained authorization engine (OpenFGA / SpiceDB / Cedar /
  AWS Verified Permissions) in the loop: pre-filter by coarse tokens, then check each candidate.
- **Permission changes must propagate.** Someone leaves a team; their access must disappear from
  retrieval too. That is a sync problem (section 6).
- **Defense in depth.** Check at the API layer and the document layer, independently.
- Log every authorization decision (user, doc, rule) for audit.
- Resolve the user's identity from your IdP token, never from a request field.

**In this repo:** `models.Principal.acl_tokens`, `store._acl_clause` (pre-filter in both BM25
and dense SQL), re-check in `retrieval.py`, `audit.py`, and the `acl_leaks` gate in `eval/run_eval.py`.

**Demo:** `examples/acl_prefilter` shows a post-filter returning nothing where the in-query filter returns the allowed doc.

---

## 6. Freshness and deletes

Teams often build inserts and leave deletes as a TODO. Then a retired policy keeps answering
questions for months ("ghost chunks").

- Keep a content-hash ledger per document. Unchanged hash: skip (no embedding cost). Changed:
  replace all chunks for that doc atomically. Missing from source: delete.
- Tombstone first, then delete chunks, so a half-finished delete never serves stale content.
  Also check document status at read time, so if ingestion breaks for a weekend you degrade to
  "missing" rather than "wrong".
- Use stable source ids (page id, file id), not paths. A rename should be an update, not
  a delete + add.
- Event-driven sync (webhooks, CDC) for freshness, plus a periodic full reconcile because
  event streams miss things.
- Refuse a sync that would delete most of the index. A wrong path or a failed connector returns few
  docs, and the reconcile would delete everything else.
- Record which embedding model built the index. Vectors from two models do not compare, so a switch
  needs a fresh index.
- Audit it: pick 50 docs deleted last week and check none of their chunks are still retrievable.

**In this repo:** `ingest/sync.py` (including the mass-delete guard), `store.delete_document`,
`store.bind_embedder`, status join in every search,
`tests/test_core.py::test_resync_skips_unchanged_and_deletes_removed`.

**Demo:** `examples/deletes_and_sync` runs re-sync, delete, mass-delete guard and ACL change.

---

## 7. Treat retrieved text as untrusted input (indirect prompt injection)

Anyone who can edit a wiki page can put instructions in front of your model. The payload can
sit dormant for weeks until a query surfaces it. There is no equivalent of parameterized SQL
for this; it needs layers:

- **Ingest**: scan documents, flag (and route for review) anything that looks like instructions
  to an AI.
- **Context**: scan retrieved chunks again at query time, titles and headings included. Fence them
  in clear delimiters and escape any tag that could open or close the fence. Tell
  the model that text inside sources is data, not instructions. A guardrail that only checks
  the user's question is blind to this by construction.
- **Output**: check citations, check grounding, redact secrets.
- **Action**: if the system can call tools, allow-list them and never let retrieved text
  trigger a write.
- **Log** which passages produced each answer, or an injection can run unnoticed.

A 2026 thesis comparing defenses found rule-based sanitization was the strongest single layer
and the full stack did best. Regex is a first line only; add a classifier (Prompt Guard /
Llama Guard, a managed guardrail service, or an LLM judge) behind the same interface.

**In this repo:** `guardrails.py` (input, context, output layers), ingest flagging in `sync.py`,
the planted attack in `sample_docs/wiki/team-offsite-notes.md`.

**Demo:** `examples/injection` flags two attacks at ingest, drops them at query time, checks that a third cannot break the source fence, and prints the obfuscated one a regex misses.

---

## 8. Grounded generation with citations and an explicit refusal

- Answer only from sources, cite every claim, and give a fixed refusal string when the answer
  is not there. A fixed string lets you count refusals and test for them.
- Validate citations after generation (do the ids exist?). Uncited claims are a hallucination signal.
- Count an answer as a refusal only when it equals the refusal string. A longer answer that quotes it
  is still an answer, and keeps its sources.
- When sources conflict, say so and prefer the newer one. Put `updated_at` in the source header.
- Top-k always returns *something*. Without a relevance threshold (reranker score) or an LLM
  that refuses, irrelevant chunks get presented as answers. You can see this in the demo.

**In this repo:** `pipeline.SYSTEM`, `guardrails.check_output`, `RAGKIT_MIN_RELEVANCE`.

**Demo:** `examples/refusal` tests how ragkit handles a refusal, with a scripted model and optionally a real one.

---

## 9. Agentic RAG: route hard questions to it

Agentic RAG (the model plans queries, retrieves several times, judges results, retries) handles
multi-hop and comparison questions better. It also costs 5 to 50x more and takes seconds
to tens of seconds. The pattern that keeps showing up in production is classic RAG by default,
with a router that sends complex questions to the agentic path. A 2026 comparison found agentic RAG
better at query rewriting, while a well-built fixed pipeline stayed more reliable at reranking
in well-defined domains.

Start with two cheap wins before a full agent loop: condensing follow-up questions into
standalone ones, and decomposing obviously compound questions into 2 to 3 sub-queries.

**In this repo:** `rewrite.py` (`condense`, `needs_decomposition`, `decompose`), fused by the
multi-query `HybridRetriever.retrieve`. Upgrade path: LangGraph / your own loop with retrieval
as a tool.

---

## 10. Evaluate retrieval and generation separately, from day one

- **Retrieval metrics** (hit@k, recall@k, MRR) are cheap and deterministic. Most bad answers are
  retrieval failures, so this is where you look first.
- **Generation metrics** via RAGAS: faithfulness, answer relevancy, context precision, context
  recall. Useful as trend lines, but LLM-judge scores can be unreliable (one 2026 paper found
  RAGAS faithfulness noisy on mid-range hardware). Spot-check by hand.
- **Security metrics**: ACL leak count must be zero. Gate CI on it. The workflow in this repo fails on
  any leak.
- Build the golden set from real employee questions (search logs, Slack, tickets), including
  questions that should be refused and questions asked by users without access.

**In this repo:** `eval/run_eval.py`, `eval/golden.jsonl` (includes ACL-negative cases),
`.github/workflows/ci.yml`.

---

## 11. Observability

Log per request: user, groups, rewritten queries, retrieved chunk ids with BM25/dense ranks and
scores, guardrail events, refusal, latency. This gives you an audit trail for security and a
debugging trail for quality. In production, emit it as OpenTelemetry spans (retrieve, rerank,
generate) so you can see where latency and failures come from.

**In this repo:** `audit.py` writes JSONL. It lists every retrieved chunk, including for refusals and
for chunks the injection filter dropped, and it records failed requests with an `error:` event. Wrap
pipeline stages in OTel spans next.

---

## Sources

- [RAG best practices for enterprise AI teams (TechTarget)](https://www.techtarget.com/searchenterpriseai/tip/RAG-best-practices-for-enterprise-AI-teams)
- [Enterprise RAG implementation guide (EurosHub)](https://www.euroshub.com/blogs/enterprise-rag-implementation-guide)
- [Setting up RAG for your enterprise (Lucent Innovation)](https://www.lucentinnovation.com/resources/technology-posts/setting-up-rag-for-your-enterprise)
- [Why vector search alone isn't enough: hybrid retrieval for RAG (InfoQ)](https://infoq.com/articles/vector-search-hybrid-retrieval-rag)
- [Hybrid search for RAG: BM25 + dense (Denser)](https://denser.ai/blog/hybrid-search-for-rag/)
- [Hybrid search: BM25, SPLADE and vector search combined (PremAI)](https://blog.premai.io/hybrid-search-for-rag-bm25-splade-and-vector-search-combined)
- [Hybrid search and reranking: BM25, vectors, cross-encoders (WZ-IT)](https://wz-it.com/en/knowledge/ki/hybrid-search-reranking/)
- [Production-ready RAG: hybrid search, reranking, memory (Coding Shuttle)](https://www.codingshuttle.com/blogs/building-production-ready-rag-hybrid-search-reranking-and-conversational-memory)
- [I switched from pure vector to hybrid retrieval (DEV)](https://dev.to/vapmail16/i-switched-from-pure-vector-search-to-hybrid-retrieval-in-my-rag-system-heres-what-changed-d16)
- [Contextual Retrieval (Anthropic)](https://www.anthropic.com/news/contextual-retrieval)
- [Reconstructing Context: late chunking vs contextual retrieval (arXiv 2504.19754)](https://arxiv.org/html/2504.19754v1)
- [Evaluating chunking strategies on academic texts (arXiv 2607.01852)](https://arxiv.org/pdf/2607.01852)
- [Document-level RBAC for RAG pipelines (Truto)](https://www.truto.one/blog/how-to-maintain-document-level-rbac-in-enterprise-rag-pipelines/)
- [Secure multi-tenant RAG with Bedrock and Verified Permissions (AWS)](https://aws.amazon.com/blogs/architecture/secure-multi-tenant-rag-with-amazon-bedrock-and-verified-permissions/)
- [Permission-aware retrieval must live in the vector layer (Tian Pan)](https://tianpan.co/blog/2026-05-04-permission-aware-retrieval-enterprise-rag-access-control)
- [ReBAC as the missing permission layer for RAG (NHI Mgmt Group)](https://nhimg.org/articles/rebac-is-the-missing-permission-layer-for-enterprise-rag-pipelines/)
- [When to use document-level permissions in RAG (NHI Mgmt Group)](https://nhimg.org/faq/when-should-organisations-use-document-level-permissions-in-rag/)
- [Role-based contextual isolation in RAG (Rubin Lake)](https://rubinlake.com/en/technology-radar/ai-security-and-governance/role-based-contextual-isolation-rag)
- [Retrieval cascade failure: how deletion poisons RAG (Tian Pan)](https://tianpan.co/blog/2026/05/09/retrieval-cascade-failure-document-deletion-rag)
- [Keeping RAG data fresh with incremental sync (Paragon)](https://www.useparagon.com/blog/rag-freshness-incremental-sync-deduplication)
- [Incremental updates for the knowledge base (DEV)](https://dev.to/wonderlab/rag-series-19-incremental-updates-keeping-the-knowledge-base-fresh-596c)
- [Defending against indirect prompt injection in RAG (Barking Iguana)](https://barkingiguana.com/writing/defending-against-indirect-prompt-injection-in-rag/)
- [Mitigation strategies against indirect prompt injection in tool-using RAG agents (UCLA)](https://escholarship.org/uc/item/8zq0952f)
- [Your RAG pipeline is one prompt away from a jailbreak (Superagent)](https://www.superagent.sh/blog/your-rag-pipeline-is-one-prompt-away-from-a-jailbreak)
- [Agentic RAG vs classic RAG (Blck Alpaca)](https://blckalpaca.at/en/knowledge-base/ai-agents/what-is-a-rag-system/agentic-rag-vs-classic-rag)
- [What is agentic RAG, 2026 explainer (AI Agent Rank)](https://aiagentrank.io/blog/what-is-agentic-rag-2026)
- [Is agentic RAG worth it? (arXiv 2601.07711)](https://api.emergentmind.com/papers/2601.07711)
