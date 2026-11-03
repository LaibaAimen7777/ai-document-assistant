
import pytest

from src.text_splitter import split_text


def test_split_text_returns_multiple_chunks():
    text = "A" * 1200

    chunks = split_text(text, chunk_size=500, overlap=100)

    assert len(chunks) == 3


def test_chunks_have_overlap():
    text = "A" * 1200

    chunks = split_text(text, chunk_size=500, overlap=100)

    assert chunks[0][-100:] == chunks[1][:100]


def test_empty_text_returns_empty_list():
    assert split_text("   ") == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        split_text("Hello", chunk_size=0)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("Hello", chunk_size=100, overlap=100)
