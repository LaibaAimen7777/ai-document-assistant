
from src.embeddings import generate_embeddings
from src.vector_store import get_vector_store, search_documents


def retrieve_relevant_chunks(
    query,
    persist_directory="chroma_db",
    collection_name="document_chunks",
    n_results=3
):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    # Convert the user's question into an embedding
    query_embedding = generate_embeddings([query])[0]

    # Access the existing document collection
    collection = get_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    # Find the most relevant chunks
    results = search_documents(
        collection=collection,
        query_embedding=query_embedding,
        n_results=n_results
    )

    return results
