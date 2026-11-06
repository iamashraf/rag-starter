"""Source connectors. Each one yields SourceDocument with a STABLE id and its ACL.

The local folder loader reads YAML front matter for ACLs so you can demo
permission-aware retrieval without a real Confluence/SharePoint. In production,
write one loader per source system and copy the ACL from that system
(Confluence space + page restrictions, Drive sharing, Git repo teams).
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import yaml

from ragkit.models import SourceDocument


class Loader(Protocol):
    def load(self) -> Iterator[SourceDocument]: ...


def _split_front_matter(raw: str) -> tuple[dict, str]:
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) == 3:
            meta = yaml.safe_load(parts[1]) or {}
            return meta, parts[2].lstrip("\n")
    return {}, raw


class LocalFolderLoader:
    """Loads .md / .txt (and .pdf with the [pdf] extra) from a folder.

    Front matter example:
        ---
        title: Expense policy
        acl: [group:finance, group:hr]
        owner: finance-ops
        ---
    Docs without an `acl` key get `default_acl`, which is [] (nobody can read them) unless you pass one.
    """

    def __init__(self, root: str | Path, default_acl: list[str] | None = None):
        self.root = Path(root)
        self.default_acl = default_acl if default_acl is not None else []

    def load(self) -> Iterator[SourceDocument]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"not a folder: {self.root}")
        for path in sorted(self.root.rglob("*")):
            if path.is_dir() or path.name.startswith("."):
                continue
            suffix = path.suffix.lower()
            if suffix in {".md", ".markdown", ".txt"}:
                raw = path.read_text(encoding="utf-8", errors="replace")
            elif suffix == ".pdf":
                raw = _read_pdf(path)
            else:
                continue
            meta, body = _split_front_matter(raw)
            rel = path.relative_to(self.root).as_posix()
            yield SourceDocument(
                doc_id=meta.get("id") or "local:" + hashlib.sha1(rel.encode()).hexdigest()[:16],
                title=meta.get("title") or path.stem.replace("-", " ").replace("_", " ").title(),
                text=body,
                source="local",
                uri=meta.get("uri") or f"file://{path.resolve()}",
                acl=list(meta.get("acl") or self.default_acl),
                metadata={k: v for k, v in meta.items() if k not in {"id", "title", "acl", "uri"}}
                | {"path": rel, "format": suffix.lstrip(".")},
                updated_at=str(meta.get("updated_at")) if meta.get("updated_at") else None,
            )


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Install the [pdf] extra to ingest PDFs") from e
    reader = PdfReader(str(path))
    # Page markers help citations point at a page later.
    return "\n\n".join(f"<!-- page {i + 1} -->\n{p.extract_text() or ''}" for i, p in enumerate(reader.pages))
