
def split_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks.

    Args:
        text: Document text.
        chunk_size: Maximum characters per chunk.
        overlap: Number of shared characters between chunks.

    Returns:
        A list of text chunks.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks
