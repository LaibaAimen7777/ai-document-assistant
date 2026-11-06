
import numpy as np
import pytest
import chromadb

from src.vector_store import (
    get_vector_store,
    add_documents,
    search_documents
)


@pytest.fixture
def collection(tmp_path):
    client = chromadb.PersistentClient(
        path=str(tmp_path / "test_db")
    )

    return client.create_collection(
        name="test_documents",
        metadata={"hnsw:space": "cosine"}
    )


def test_add_documents(collection):
    chunks = ["Python is a programming language."]

    embeddings = np.array([[0.1, 0.2, 0.3]])

    add_documents(collection, chunks, embeddings)

    assert collection.count() == 1


def test_search_documents(collection):
    chunks = [
        "Python is a programming language.",
        "The weather is sunny today."
    ]

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])

    add_documents(collection, chunks, embeddings)

    results = search_documents(
        collection,
        np.array([1.0, 0.0, 0.0]),
        n_results=1
    )

    assert len(results) == 1
    assert results[0]["text"] == chunks[0]


def test_empty_collection_returns_empty_list(collection):
    results = search_documents(
        collection,
        np.array([1.0, 0.0, 0.0])
    )

    assert results == []


def test_mismatched_chunks_and_embeddings(collection):
    with pytest.raises(ValueError):
        add_documents(
            collection,
            ["First chunk", "Second chunk"],
            np.array([[0.1, 0.2]])
        )
