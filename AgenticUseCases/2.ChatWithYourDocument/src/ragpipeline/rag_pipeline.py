from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from langchain_core.runnables import RunnableLambda

from .evaluation import EvaluationResult, evaluate_answer
from .generation import build_generation_chain
from .retrieval import retrieve_documents_with_scores
from embeddingpipeline.embedding import (
    DEFAULT_COLLECTION,
)


def _print_evaluation_log(evaluation: EvaluationResult) -> None:

    message = (
        f"RAG evaluation::: "
        f"relevance={evaluation.relevance_score:.3f}, "
        f"groundedness={evaluation.groundedness_score:.3f}"
    )
    logging.getLogger(__name__).info(message)
    print(message, flush=True)


def build_rag_chain(
    source_paths: Sequence[str | Path],
    llm_model: str | None = None,
    embedding_model: str | None = None,
    base_url: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
):
    from .retrieval import get_vector_store

    vector_store = get_vector_store(source_paths, collection_name=collection_name)
    generation_chain = build_generation_chain(llm_model=llm_model, base_url=base_url)

    def answer_with_scores(question: str) -> str:
        documents, relevance_scores = retrieve_documents_with_scores(
            question,
            source_paths,
            collection_name=collection_name,
            top_k=4,
        )
        answer = generation_chain.invoke({"context": documents, "question": question})
        evaluation = evaluate_answer(answer, documents, relevance_scores)
        _print_evaluation_log(evaluation)
        return answer

    rag_chain = RunnableLambda(answer_with_scores)
    return rag_chain, vector_store


def answer_question(
    question: str,
    source_paths: Sequence[str | Path],
    llm_model: str | None = None,
    embedding_model: str | None = None,
    base_url: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
):
    if not question or not question.strip():
        raise ValueError("Please provide a valid question.")

    retrieval_chain, _ = build_rag_chain(
        source_paths,
        llm_model=llm_model,
        embedding_model=embedding_model,
        base_url=base_url,
        collection_name=collection_name,
    )
    return retrieval_chain.invoke(question)


def answer_question_with_evaluation(
    question: str,
    source_paths: Sequence[str | Path],
    llm_model: str | None = None,
    base_url: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> tuple[str, EvaluationResult]:
    """Retrieve context, generate an answer, and evaluate the result."""
    context, relevance_scores = retrieve_documents_with_scores(
        question,
        source_paths,
        collection_name=collection_name,
        top_k=4,
    )
    generation_chain = build_generation_chain(llm_model=llm_model, base_url=base_url)
    answer = generation_chain.invoke({"context": context, "question": question})
    evaluation = evaluate_answer(answer, context, relevance_scores)
    _print_evaluation_log(evaluation)
    return answer, evaluation
