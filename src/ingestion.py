from pathlib import Path

from src.pdf_loader import extract_text_from_pdf
from src.text_splitter import split_text
from src.embeddings import generate_embeddings
from src.vector_store import get_vector_store, add_documents

def ingest_pdf(
        pdf_path,
        persistent_dictionary="chrome_db",
        collection_name="document_chunks"
):
    text=extract_text_from_pdf(pdf_path)
    chunks=split_text(text)

    if not chunks:
        raise ValueError("No text chunks were generated from the Pdf")

    embeddings=generate_embeddings(chunks)

    collection=get_vector_store(
        persist_directory=persistent_dictionary,
        collection_name=collection_name
    )

    filename=Path(pdf_path).name

    metadatas=[
        {
            "source":filename,
            "chunk_index":index
        }
        for index in range(len(chunks))
    ]

    add_documents(
        collection=collection,
        chunks=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return{
        "filename":filename,
        "total_chunks":len(chunks),
        "stored_documents":collection.count()
    }