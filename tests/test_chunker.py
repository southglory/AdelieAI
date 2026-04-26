from datetime import datetime, timezone

from core.retrieval.chunker import RecursiveTextSplitter
from core.schemas.retrieval import Document


def _doc(content: str) -> Document:
    return Document(
        id="d1",
        title="t",
        source="s",
        content=content,
        metadata={"key": "v"},
        created_at=datetime.now(timezone.utc),
    )


def test_short_text_yields_single_chunk() -> None:
    splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.split(_doc("hello world"))
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].position == 0
    assert chunks[0].doc_id == "d1"
    assert chunks[0].metadata["doc_title"] == "t"
    assert chunks[0].metadata["key"] == "v"


def test_paragraph_separator_used_first() -> None:
    splitter = RecursiveTextSplitter(chunk_size=20, chunk_overlap=0)
    text = "para1\n\npara2\n\npara3"
    chunks = splitter.split(_doc(text))
    assert len(chunks) >= 2
    joined = "".join(c.text for c in chunks)
    assert "para1" in joined and "para2" in joined and "para3" in joined


def test_chunks_respect_size_budget() -> None:
    splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=5)
    text = ("abcdef ghijkl mnopqr stuvwx yz0123 456789 " * 10).strip()
    chunks = splitter.split(_doc(text))
    for c in chunks:
        assert len(c.text) <= 60


def test_chunks_have_overlap_when_split() -> None:
    splitter = RecursiveTextSplitter(chunk_size=40, chunk_overlap=10)
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    chunks = splitter.split(_doc(text))
    assert len(chunks) >= 2
    if len(chunks) >= 2:
        end_of_first = chunks[0].text[-10:]
        start_of_second = chunks[1].text[:10]
        assert end_of_first == start_of_second or any(
            ch in chunks[1].text[:15] for ch in end_of_first.split()
        )


def test_empty_content_yields_no_chunks() -> None:
    splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.split(_doc(""))
    assert chunks == []


def test_positions_are_sequential() -> None:
    splitter = RecursiveTextSplitter(chunk_size=20, chunk_overlap=0)
    chunks = splitter.split(_doc("a\n\nb\n\nc\n\nd"))
    positions = [c.position for c in chunks]
    assert positions == list(range(len(chunks)))


def test_overlap_must_be_smaller_than_size() -> None:
    import pytest

    with pytest.raises(ValueError):
        RecursiveTextSplitter(chunk_size=100, chunk_overlap=100)
