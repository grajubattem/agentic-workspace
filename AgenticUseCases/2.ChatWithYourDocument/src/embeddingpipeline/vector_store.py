from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from .chunking import split_documents


def create_chroma_store(
    source_paths: Sequence[str | Path],
    collection_name: str,
):
    """Extract, chunk, embed, and persist documents in Chroma."""
    from .embedding import (
        extract_documents_from_paths,
        get_chroma_directory,
        get_embedding_model_config,
    )

    if not source_paths:
        raise ValueError("No document files were provided for indexing.")

    documents = extract_documents_from_paths(source_paths)
    if not documents:
        raise ValueError("No readable text was found in the uploaded files.")

    embedding_model, _ = get_embedding_model_config()
    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    return Chroma.from_documents(
        documents=split_documents(documents),
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=get_chroma_directory(),
    )


def get_existing_chroma_store(collection_name: str):
    """Open an existing persistent Chroma collection."""
    from .embedding import get_chroma_directory, get_embedding_model_config

    embedding_model, _ = get_embedding_model_config()
    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=get_chroma_directory(),
    )


def main() -> None:
    from .embedding import DEFAULT_COLLECTION, extract_documents_from_paths, get_default_source_files

    source_files = get_default_source_files()
    if not source_files:
        raise FileNotFoundError("No PDF or Excel source files were found in the project folder.")

    source_path = next(
        (path for path in source_files if path.suffix.lower() == ".pdf"),
        source_files[0],
    )
    documents = extract_documents_from_paths([source_path])
    chunks = split_documents(documents)
    store = create_chroma_store([source_path], collection_name=DEFAULT_COLLECTION)

    print(f"Source path: {source_path}")
    print(f"Extracted documents: {len(documents)}")
    print(f"Split chunks: {len(chunks)}")
    print(f"Store type: {type(store).__name__}")
    print(f"Saved {len(chunks)} chunks to Chroma collection '{DEFAULT_COLLECTION}'.")


if __name__ == "__main__":
    main()
