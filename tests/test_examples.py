import importlib.util
import sys
from pathlib import Path

import httpx
import pytest

from ragkit.ingest.loaders import LocalFolderLoader

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))
import _common  # noqa: E402
from _common import Unavailable  # noqa: E402

OFFLINE: list[str] = ["hybrid_identifier", "acl_prefilter", "injection", "deletes_and_sync", "refusal"]


def test_scripted_llm_returns_fixed_answer():
    llm = _common.ScriptedLLM("fixed")
    assert llm.complete([{"role": "user", "content": "a"}]) == "fixed"
    assert llm.complete([], max_tokens=1, temperature=0.5) == "fixed"


def test_report_prints_pass_and_fail(capsys):
    assert _common.report("n", "e", "o", True) is True
    assert "PASS" in capsys.readouterr().out
    assert _common.report("n", "e", "o", False) is False
    out = capsys.readouterr().out
    assert "FAIL" in out and "EXPECTED" in out and "OBSERVED" in out


def test_make_settings_ignores_ambient_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGKIT_EMBED_PROVIDER", "openai")
    monkeypatch.setenv("RAGKIT_RERANKER", "cross-encoder")
    monkeypatch.setenv("RAGKIT_MIN_RELEVANCE", "5")
    s = _common.make_settings(tmp_path)
    assert (s.embed_provider, s.reranker, s.min_relevance, s.contextualize) == ("hash", "none", 0.0, False)
    assert Path(s.db_path).parent == tmp_path and Path(s.audit_path).parent == tmp_path


def test_require_ollama_unreachable_raises_unavailable(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("no")

    monkeypatch.setattr(httpx, "get", boom)
    with pytest.raises(Unavailable):
        _common.require_ollama(["nomic-embed-text"])


class _Resp:
    def __init__(self, names):
        self._names = names

    def raise_for_status(self):
        pass

    def json(self):
        return {"models": [{"name": n} for n in self._names]}


def test_require_ollama_missing_model_names_pull_command(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(["llama3.1:8b"]))
    with pytest.raises(Unavailable) as e:
        _common.require_ollama(["llama3.1"])
    assert "ollama pull llama3.1" in str(e.value) and "llama3.1:8b" in str(e.value)
    _common.require_ollama(["llama3.1:8b"])
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(["nomic-embed-text:latest"]))
    _common.require_ollama(["nomic-embed-text"])


def test_cli_exit_codes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x"])

    def unavailable(_):
        raise Unavailable("nope")

    for fn, code in [(lambda _: True, 0), (lambda _: False, 1), (unavailable, 2)]:
        with pytest.raises(SystemExit) as e:
            _common.cli(fn)
        assert e.value.code == code


def test_every_demo_doc_has_acl():
    for folder in EXAMPLES.glob("*/docs"):
        for doc in LocalFolderLoader(folder).load():
            assert doc.acl, f"{doc.id} has no acl"


@pytest.mark.parametrize("name", OFFLINE)
def test_offline_demo_passes(name, monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(f"demo_{name}", EXAMPLES / name / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    assert module.main("hash") is True
    assert list(tmp_path.iterdir()) == []


def test_hybrid_identifier_dense_note_follows_embedder():
    spec = importlib.util.spec_from_file_location("demo_note", EXAMPLES / "hybrid_identifier" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "hash" in module.dense_note("hash")
    assert "hash" not in module.dense_note("ollama") and "semantic" not in module.dense_note("ollama")


def test_acl_prefilter_rejects_non_hash_embedder():
    spec = importlib.util.spec_from_file_location("demo_acl", EXAMPLES / "acl_prefilter" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(Unavailable):
        module.main("ollama")


def test_injection_rejects_non_hash_embedder():
    spec = importlib.util.spec_from_file_location("demo_inj", EXAMPLES / "injection" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(Unavailable):
        module.main("ollama")


def test_deletes_and_sync_rejects_non_hash_embedder():
    spec = importlib.util.spec_from_file_location("demo_del", EXAMPLES / "deletes_and_sync" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(Unavailable):
        module.main("ollama")
