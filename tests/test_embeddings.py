
import numpy as np
import pytest

from src import embeddings


class FakeEmbeddingModel:
    def encode(self, texts, convert_to_numpy, normalize_embeddings):
        return np.zeros((len(texts), 384))


def test_generate_embeddings(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "get_embedding_model",
        lambda: FakeEmbeddingModel()
    )

    result = embeddings.generate_embeddings(
        ["Artificial intelligence", "Machine learning"]
    )

    assert result.shape == (2, 384)


def test_empty_input():
    assert embeddings.generate_embeddings([]) == []


def test_invalid_input():
    with pytest.raises(TypeError):
        embeddings.generate_embeddings("Hello")


def test_empty_text_chunk():
    with pytest.raises(ValueError):
        embeddings.generate_embeddings(["Hello", "   "])
