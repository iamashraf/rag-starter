# Pattern demos

Each demo is one script that shows a pattern from [`docs/PATTERNS.md`](../docs/PATTERNS.md) working, and fails loudly if it stops working. Five run offline with no keys. Two use a local Ollama.

## Run a demo

```bash
pip install -e ".[dev]"
python examples/acl_prefilter/run.py
```

Every demo builds its index in a temp directory. It never touches `data/ragkit.db` or `data/audit.jsonl`.

| Exit code | Meaning |
|---|---|
| 0 | Every check passed. |
| 1 | A check failed. |
| 2 | The demo cannot run here, for example Ollama is down or a model is missing. |

## Demos

| Demo | Needs | Pattern | Shows |
|---|---|---|---|
| `hybrid_identifier` | offline | 2 | BM25 puts the exact incident id first. |
| `hybrid_paraphrase` | Ollama | 2 | Dense search finds a paraphrase BM25 cannot. |
| `acl_prefilter` | offline | 5 | A post-filter on the top-k returns nothing. A filter inside the query does not. |
| `deletes_and_sync` | offline | 6 | Re-sync is free, deletes propagate, a mass delete is refused, an ACL edit moves visibility. |
| `injection` | offline | 7 | Injected text is flagged at ingest, dropped at query time, and cannot break the source fence. |
| `refusal` | offline, plus optional Ollama | 8 | The refusal string becomes `refused=True`. With `--ollama` it also prints what a real model did. |

`hybrid_paraphrase` needs `--ollama`. Run the others with `--ollama` to use Ollama embeddings where the demo supports it. The offline-only demos exit 2 if you do.

## Read the output

Each check prints three lines:

```
EXPECTED  what should happen
OBSERVED  what happened
PASS      (or FAIL)
```

Two other labels never change the exit code:

- **observation**: something the demo reports but does not assert, such as a real model's reply.
- **known gap**: a case ragkit does not handle. The injection demo prints one.

The `refusal` demo with a scripted LLM tests how ragkit handles a model's output. It does not test a model.

## Run with Ollama

```bash
podman compose -f compose.ollama.yaml up -d      # or: docker compose ...
podman compose -f compose.ollama.yaml exec ollama ollama pull nomic-embed-text
podman compose -f compose.ollama.yaml exec ollama ollama pull llama3.1:8b
export RAGKIT_LLM_MODEL=llama3.1:8b
python examples/hybrid_paraphrase/run.py --ollama
python examples/refusal/run.py --ollama
```

- **Use the exact tag.** `llama3.1` is not `llama3.1:8b`. A bare name only matches `:latest`. The demos exit 2 and print the `ollama pull` command when the name does not match.
- **One Ollama per volume.** To reuse a volume you already have, set `RAGKIT_OLLAMA_VOLUME=<name>` in `.env`. Stop any other container on that volume first, because two pulls at once can corrupt it.
- **Never run `down -v` on a shared volume.** It deletes your models.
- **The port is published on localhost only.** Ollama has no authentication.

## Measured results

Run on 2026-11-07 with Ollama 0.40.11 and `nomic-embed-text`, against throwaway databases. The corpus is `examples/hybrid_paraphrase/docs`. The questions are in `golden.jsonl`.

| Setup | hit@1 | hit@3 | hit@5 | MRR@5 |
|---|---|---|---|---|
| `hash` embedder | 0.167 | 0.333 | 0.333 | 0.222 |
| Ollama `nomic-embed-text` | 0.667 | 0.833 | 1.0 | 0.783 |

Both setups had 0 permission leaks. To reproduce it:

```bash
ragkit ingest examples/hybrid_paraphrase/docs
python eval/run_eval.py --golden examples/hybrid_paraphrase/golden.jsonl
```

Read these numbers with care:

- **It is a best case.** We wrote the 12 docs and 6 questions so the questions share no stemmed word with their target. It is not a benchmark.
- **Fusion can lower a rank.** For two questions the fused rank was 2 and 5, not 1. BM25 matched common words such as "after" and "off" in other docs.
- **One model run is one observation.** In one run, `llama3.1:8b` replied to the unanswerable question with the exact refusal string. Another run or model can differ, so the demo does not assert it.

## Known limits

- The obfuscated instruction in the injection demo passes the regex scanner. `PATTERNS.md` recommends a classifier for that.
- The `hash` embedder is not semantic. No offline demo claims semantic behavior.
- Verified only on a Mac with Podman. CI covers the offline demos on Python 3.11 and 3.15. Other platforms are untested.

## Add a demo

1. Put it in `examples/<name>/` with a `docs/` folder and a `run.py` that exposes `main(embedder="hash") -> bool`.
2. Build the index in a temp directory with `_common.build`. Give every doc an explicit `id` and `acl`.
3. Assert the claim the demo makes, and break the behavior once to confirm the demo prints `FAIL`.
4. Add the name to `OFFLINE` in `tests/test_examples.py` if it runs offline.
