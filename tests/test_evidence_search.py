"""Focused contract tests for corpus-backed evidence retrieval."""

from core.tools.evidence_search import EvidenceSearch, FileCorpusEvidenceSearch


def test_search_reads_and_ranks_real_corpus_files(tmp_path) -> None:
    (tmp_path / "glass.md").write_text(
        "유리 조각은 방 안쪽에서 발견됐다. 유리는 창문 아래에도 있었다.",
        encoding="utf-8",
    )
    (tmp_path / "alibi.txt").write_text("용의자는 가게에 있었다.", encoding="utf-8")

    result = EvidenceSearch(corpus_dir=tmp_path).call({"query": "유리"})

    assert result["n_hits"] == 1
    assert result["hits"][0]["path"] == "glass.md"
    assert result["hits"][0]["score"] == 2.0
    assert "방 안쪽" in result["hits"][0]["snippet"]
    assert result["backend"] == "filesystem_keyword"
    assert result["source"]["files_indexed"] == 2


def test_search_returns_metadata_with_no_hits(tmp_path) -> None:
    (tmp_path / "case.md").write_text("기록된 증거는 유리뿐이다.", encoding="utf-8")

    result = EvidenceSearch(corpus_dir=tmp_path).call({"query": "마시멜로"})

    assert result["n_hits"] == 0
    assert result["hits"] == []
    assert result["backend"] == "filesystem_keyword"
    assert result["source"]["path"] == str(tmp_path)


def test_missing_corpus_returns_actionable_error(tmp_path) -> None:
    missing = tmp_path / "not-created"

    result = EvidenceSearch(corpus_dir=missing).call({"query": "유리"})

    assert result["n_hits"] == 0
    assert result["hits"] == []
    assert str(missing) in result["error"]
    assert "Create the directory" in result["error"]
    assert result["source"] == {"type": "unavailable"}


def test_tool_accepts_a_replaceable_search_backend() -> None:
    class RecordingBackend(FileCorpusEvidenceSearch):
        backend_name = "recording_test_backend"

    # The concrete adapter is injected through the synchronous port.
    tool = EvidenceSearch(backend=RecordingBackend())

    result = tool.call({"query": "유리"})

    assert result["backend"] == "recording_test_backend"
    assert result["n_hits"] >= 1
