from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from langchain_core.documents import Document


@dataclass(frozen=True)
class EvaluationResult:
    answer_present: bool
    context_used: bool
    relevance_score: float
    groundedness_score: float

    @property
    def passed(self) -> bool:
        return self.answer_present and self.context_used


def evaluate_answer(
    answer: str,
    context: Iterable[Document],
    relevance_scores: Iterable[float] | None = None,
) -> EvaluationResult:
    """Run lightweight local checks on a generated RAG answer."""
    documents = list(context)
    context_text = " ".join(document.page_content.lower() for document in documents)
    answer_terms = {
        term
        for term in re.findall(r"[a-zA-Z0-9]+", answer.lower())
        if len(term) > 2
    }
    grounded_terms = {term for term in answer_terms if term in context_text}
    groundedness_score = len(grounded_terms) / len(answer_terms) if answer_terms else 0.0
    scores = list(relevance_scores or [])
    relevance_score = sum(scores) / len(scores) if scores else 0.0

    return EvaluationResult(
        answer_present=bool(answer and answer.strip()),
        context_used=bool(documents),
        relevance_score=round(max(0.0, min(1.0, relevance_score)), 3),
        groundedness_score=round(max(0.0, min(1.0, groundedness_score)), 3),
    )
