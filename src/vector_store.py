
import chromadb


def get_vector_store(
    persist_directory="chroma_db",
    collection_name="document_chunks"
):
    """
    Create or retrieve a persistent ChromaDB collection.
    """

    client = chromadb.PersistentClient(
        path=persist_directory
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def add_documents(collection, chunks, embeddings, metadatas=None):
    """
    Store text chunks and their embeddings in ChromaDB.
    """

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings must match."
        )

    if not chunks:
        return

    if metadatas is None:
        metadatas = [{"chunk_index":i}
                     for i in range(len(chunks))]

    if len(metadatas) != len(chunks):
        raise ValueError(
            "Number of metadata entries must match chunks."
        )

    existing_count = collection.count()

    ids = [
        f"chunk_{existing_count + i}"
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )


def search_documents(collection, query_embedding, n_results=3):
    """
    Retrieve the most similar document chunks.
    """

    if n_results <= 0:
        raise ValueError("n_results must be greater than zero.")

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(n_results, collection.count())
    )

    return [
        {
            "text": document,
            "metadata": metadata,
            "distance": distance
        }
        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]
