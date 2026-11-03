
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def get_embedding_model():
    """
    Load the embedding model only when needed.
    """
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def generate_embeddings(texts):
    """
    Convert a list of text chunks into numerical embeddings.

    Args:
        texts: List of text strings.

    Returns:
        A NumPy array containing the embeddings.
    """

    if not isinstance(texts, list):
        raise TypeError("Input must be a list of strings.")

    if not all(isinstance(text, str) for text in texts):
        raise TypeError("Every item must be a string.")

    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Text chunks cannot be empty.")

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings
