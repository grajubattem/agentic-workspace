from __future__ import annotations

from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from embeddingpipeline.embedding import (
    DEFAULT_COLLECTION,
    get_or_create_chroma_store,
)


def get_vector_store(
    source_paths: Sequence[str | Path],
    collection_name: str = DEFAULT_COLLECTION,
) -> VectorStore:
    """Load an existing Chroma collection or build it from source files."""
    return get_or_create_chroma_store(source_paths, collection_name=collection_name)


def create_retriever(
    source_paths: Sequence[str | Path],
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 4,
):
    """Create a retriever configured for the requested number of results."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    vector_store = get_vector_store(source_paths, collection_name=collection_name)
    return vector_store.as_retriever(search_kwargs={"k": top_k}), vector_store


def retrieve_documents(
    question: str,
    source_paths: Sequence[str | Path],
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 4,
) -> list[Document]:
    """Retrieve the most relevant chunks for a question."""
    if not question or not question.strip():
        raise ValueError("Please provide a valid question.")

    retriever, _ = create_retriever(
        source_paths,
        collection_name=collection_name,
        top_k=top_k,
    )
    return retriever.invoke(question)


def retrieve_documents_with_scores(
    question: str,
    source_paths: Sequence[str | Path],
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 4,
) -> tuple[list[Document], list[float]]:
    """Retrieve chunks and Chroma relevance scores for evaluation logging."""
    if not question or not question.strip():
        raise ValueError("Please provide a valid question.")

    vector_store = get_vector_store(source_paths, collection_name=collection_name)
    scored_documents = vector_store.similarity_search_with_relevance_scores(
        question,
        k=top_k,
    )
    documents = [document for document, _ in scored_documents]
    scores = [float(score) for _, score in scored_documents]
    return documents, scores
