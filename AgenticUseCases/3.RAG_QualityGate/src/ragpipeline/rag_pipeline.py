from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from evaluation.evaluation import EvaluationResult, evaluate_answer
from evaluation.ragas_judge import (
    RagasScores,
    empty_scores,
    judge_enabled,
    print_scores,
    score_answer,
)
from .generation import build_generation_chain
from .retrieval import retrieve_documents_with_scores
from embeddingpipeline.embedding import (
    DEFAULT_COLLECTION,
)

DEFAULT_TOP_K = 4


@dataclass(frozen=True)
class RagResponse:
    """An answer plus the retrieval context and quality-gate scores behind it."""

    answer: str
    documents: list[Document] = field(default_factory=list)
    relevance_scores: list[float] = field(default_factory=list)
    evaluation: EvaluationResult | None = None
    ragas: RagasScores = field(default_factory=empty_scores)


def _print_evaluation_log(evaluation: EvaluationResult) -> None:

    message = (
        f"RAG evaluation::: "
        f"relevance={evaluation.relevance_score:.3f}, "
        f"groundedness={evaluation.groundedness_score:.3f}"
    )
    logging.getLogger(__name__).info(message)
    print(message, flush=True)


def answer_with_quality_gate(
    question: str,
    source_paths: Sequence[str | Path],
    llm_model: str | None = None,
    base_url: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = DEFAULT_TOP_K,
    reference: str | None = None,
    run_ragas: bool | None = None,
    judge_model: str | None = None,
) -> RagResponse:
    """Answer a question and score it with the local checks and the RAGAS judge.

    ``reference`` is an optional ground-truth answer. Context precision and context
    recall need one; without it the generated answer is used as a proxy reference and
    the scores are reported as ``self-referenced``.
    """
    if not question or not question.strip():
        raise ValueError("Please provide a valid question.")

    documents, relevance_scores = retrieve_documents_with_scores(
        question,
        source_paths,
        collection_name=collection_name,
        top_k=top_k,
    )
    generation_chain = build_generation_chain(llm_model=llm_model, base_url=base_url)
    answer = generation_chain.invoke({"context": documents, "question": question})

    evaluation = evaluate_answer(answer, documents, relevance_scores)
    _print_evaluation_log(evaluation)

    should_judge = judge_enabled() if run_ragas is None else run_ragas
    if should_judge:
        print("Running RAGAS LLM-as-judge evaluation...", flush=True)
        ragas_scores = score_answer(
            question,
            answer,
            documents,
            reference=reference,
            judge_model=judge_model,
            base_url=base_url,
        )
    else:
        ragas_scores = empty_scores("disabled")
    print_scores(ragas_scores)

    return RagResponse(
        answer=answer,
        documents=list(documents),
        relevance_scores=list(relevance_scores),
        evaluation=evaluation,
        ragas=ragas_scores,
    )


def build_rag_chain(
    source_paths: Sequence[str | Path],
    llm_model: str | None = None,
    embedding_model: str | None = None,
    base_url: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
):
    from .retrieval import get_vector_store

    vector_store = get_vector_store(source_paths, collection_name=collection_name)

    def answer_with_scores(question: str) -> str:
        response = answer_with_quality_gate(
            question,
            source_paths,
            llm_model=llm_model,
            base_url=base_url,
            collection_name=collection_name,
        )
        return response.answer

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
    reference: str | None = None,
) -> tuple[str, EvaluationResult, RagasScores]:
    """Retrieve context, generate an answer, and evaluate the result."""
    response = answer_with_quality_gate(
        question,
        source_paths,
        llm_model=llm_model,
        base_url=base_url,
        collection_name=collection_name,
        reference=reference,
    )
    return response.answer, response.evaluation, response.ragas
