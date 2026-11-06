"""Index storage: one SQLite file holding docs, chunks, ACLs, BM25 (FTS5) and vectors.

Why SQLite for a starter: zero infrastructure, and it lets you see every moving
part. The interface (upsert/delete/search_bm25/search_dense) maps directly onto
Postgres + pgvector + tsvector/ParadeDB, OpenSearch, Qdrant, Weaviate or Elastic
when you outgrow it (dense search here is brute force, fine to ~100k chunks).

Two rules this store enforces that most tutorials skip:
  1. ACL filtering happens INSIDE the search query (pre-filter), not on the top-k
     afterwards. Post-filtering leaks less obviously but starves results: the top 30
     may all be restricted, leaving the user with nothing or with worse chunks.
  2. A document has a status. Deletion tombstones the doc first so it stops being
     served immediately, then removes chunks. Search joins on status='active', so a
     half-finished delete can never surface a ghost chunk.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np

from ragkit.models import Chunk, SourceDocument

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  title TEXT NOT NULL,
  uri TEXT NOT NULL,
  acl TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  updated_at TEXT,
  indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
  metadata TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
  rowid INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_id TEXT UNIQUE NOT NULL,
  doc_id TEXT NOT NULL REFERENCES documents(doc_id),
  ordinal INTEGER NOT NULL,
  text TEXT NOT NULL,
  embed_text TEXT NOT NULL,
  heading_path TEXT,
  metadata TEXT,
  embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS chunk_acl (
  chunk_id TEXT NOT NULL,
  token TEXT NOT NULL,
  PRIMARY KEY (chunk_id, token)
);
CREATE INDEX IF NOT EXISTS idx_acl_token ON chunk_acl(token);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  embed_text, content='chunks', content_rowid='rowid', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, embed_text) VALUES (new.rowid, new.embed_text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, embed_text) VALUES ('delete', old.rowid, old.embed_text);
END;
"""

_FTS_TOKEN = re.compile(r"[\w\-\.]+", re.UNICODE)


class Store:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # ---------- ledger ----------
    def get_hash(self, doc_id: str) -> str | None:
        row = self.db.execute("SELECT content_hash FROM documents WHERE doc_id=? AND status='active'", (doc_id,)).fetchone()
        return row["content_hash"] if row else None

    def active_doc_ids(self, source: str) -> set[str]:
        rows = self.db.execute("SELECT doc_id FROM documents WHERE source=? AND status='active'", (source,))
        return {r["doc_id"] for r in rows}

    # ---------- writes ----------
    def upsert_document(self, doc: SourceDocument, content_hash: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        with self.db:
            self._delete_chunks(doc.doc_id)
            self.db.execute(
                """INSERT INTO documents(doc_id, source, title, uri, acl, content_hash, status, updated_at, metadata)
                   VALUES (?,?,?,?,?,?, 'active', ?, ?)
                   ON CONFLICT(doc_id) DO UPDATE SET source=excluded.source, title=excluded.title, uri=excluded.uri,
                     acl=excluded.acl, content_hash=excluded.content_hash, status='active',
                     updated_at=excluded.updated_at, metadata=excluded.metadata, indexed_at=CURRENT_TIMESTAMP""",
                (doc.doc_id, doc.source, doc.title, doc.uri, json.dumps(doc.acl), content_hash, doc.updated_at,
                 json.dumps(doc.metadata, default=str)),
            )
            for c, v in zip(chunks, vectors):
                self.db.execute(
                    """INSERT INTO chunks(chunk_id, doc_id, ordinal, text, embed_text, heading_path, metadata, embedding)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (c.chunk_id, c.doc_id, c.ordinal, c.text, c.embed_text, c.heading_path,
                     json.dumps(c.metadata, default=str), np.asarray(v, dtype=np.float32).tobytes()),
                )
                self.db.executemany(
                    "INSERT OR IGNORE INTO chunk_acl(chunk_id, token) VALUES (?,?)", [(c.chunk_id, t) for t in c.acl]
                )

    def delete_document(self, doc_id: str) -> None:
        # Step 1: tombstone so it stops being served right now.
        with self.db:
            self.db.execute("UPDATE documents SET status='deleted' WHERE doc_id=?", (doc_id,))
        # Step 2: physically remove chunks (safe to retry).
        with self.db:
            self._delete_chunks(doc_id)

    def _delete_chunks(self, doc_id: str) -> None:
        ids = [r["chunk_id"] for r in self.db.execute("SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,))]
        if ids:
            self.db.executemany("DELETE FROM chunk_acl WHERE chunk_id=?", [(i,) for i in ids])
            self.db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))

    # ---------- reads ----------
    @staticmethod
    def _acl_clause(tokens: set[str]) -> tuple[str, list[str]]:
        ph = ",".join("?" * len(tokens))
        return f"EXISTS (SELECT 1 FROM chunk_acl a WHERE a.chunk_id=c.chunk_id AND a.token IN ({ph}))", sorted(tokens)

    def search_bm25(self, query: str, acl_tokens: set[str], limit: int) -> list[tuple[str, float]]:
        terms = [t for t in _FTS_TOKEN.findall(query.lower()) if len(t) > 1]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in terms)  # quote every term: no FTS syntax injection
        acl_sql, acl_args = self._acl_clause(acl_tokens)
        rows = self.db.execute(
            f"""SELECT c.chunk_id, bm25(chunks_fts) AS score
                FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE chunks_fts MATCH ? AND d.status='active' AND {acl_sql}
                ORDER BY score LIMIT ?""",
            [fts_query, *acl_args, limit],
        ).fetchall()
        return [(r["chunk_id"], -r["score"]) for r in rows]  # FTS5 bm25 is lower-is-better

    def search_dense(self, qvec: np.ndarray, acl_tokens: set[str], limit: int) -> list[tuple[str, float]]:
        acl_sql, acl_args = self._acl_clause(acl_tokens)
        rows = self.db.execute(
            f"""SELECT c.chunk_id, c.embedding FROM chunks c JOIN documents d ON d.doc_id=c.doc_id
                WHERE d.status='active' AND c.embedding IS NOT NULL AND {acl_sql}""",
            acl_args,
        ).fetchall()
        if not rows:
            return []
        mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        sims = mat @ qvec.astype(np.float32)
        idx = np.argsort(-sims)[:limit]
        return [(rows[i]["chunk_id"], float(sims[i])) for i in idx]

    def fetch(self, chunk_ids: list[str]) -> dict[str, dict]:
        if not chunk_ids:
            return {}
        ph = ",".join("?" * len(chunk_ids))
        rows = self.db.execute(
            f"""SELECT c.*, d.title, d.uri, d.acl, d.status FROM chunks c JOIN documents d ON d.doc_id=c.doc_id
                WHERE c.chunk_id IN ({ph})""",
            chunk_ids,
        ).fetchall()
        return {r["chunk_id"]: dict(r) for r in rows}

    def stats(self) -> dict:
        q = lambda s: self.db.execute(s).fetchone()[0]  # noqa: E731
        return {
            "documents_active": q("SELECT COUNT(*) FROM documents WHERE status='active'"),
            "documents_deleted": q("SELECT COUNT(*) FROM documents WHERE status='deleted'"),
            "chunks": q("SELECT COUNT(*) FROM chunks"),
        }
