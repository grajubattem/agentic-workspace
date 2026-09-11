from __future__ import annotations

import asyncio
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Sequence

from langchain_ollama import ChatOllama, OllamaEmbeddings

# ragas calls nest_asyncio.apply() while importing ragas.executor. On Python 3.12+
# that patch leaves asyncio.current_task() returning None, so asyncio.wait_for inside
# the ragas metrics raises "Timeout should be used inside a task". This module always
# runs metrics on an event loop it owns, so the patch is unnecessary here: disable it
# before ragas is imported for the first time.
try:  # pragma: no cover - depends on the installed ragas dependencies
    import nest_asyncio

    nest_asyncio.apply = lambda *args, **kwargs: None
except ImportError:
    pass

from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

DEFAULT_JUDGE_MODEL = "gemma3:4b"
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 300

METRIC_ORDER = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}

METRIC_DESCRIPTIONS = {
    "faithfulness": "Is the answer grounded in the retrieved context (no hallucinations)?",
    "answer_relevancy": "Does the answer directly address the question that was asked?",
    "context_precision": "Are the most relevant chunks ranked highest (signal-to-noise)?",
    "context_recall": "Did retrieval capture everything needed to answer the question?",
}


@dataclass(frozen=True)
class RagasScores:
    """LLM-as-judge scores for a single question/answer pair."""

    scores: dict[str, float | None]
    reference_mode: str
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def faithfulness(self) -> float | None:
        return self.scores.get("faithfulness")

    @property
    def answer_relevancy(self) -> float | None:
        return self.scores.get("answer_relevancy")

    @property
    def context_precision(self) -> float | None:
        return self.scores.get("context_precision")

    @property
    def context_recall(self) -> float | None:
        return self.scores.get("context_recall")

    @property
    def available(self) -> bool:
        return any(value is not None for value in self.scores.values())

    def as_dict(self) -> dict[str, float | None]:
        return {name: self.scores.get(name) for name in METRIC_ORDER}


def empty_scores(reason: str = "disabled") -> RagasScores:
    """Placeholder scores for when the judge is switched off or cannot run."""
    return RagasScores({name: None for name in METRIC_ORDER}, reason)


def judge_enabled() -> bool:
    """Allow the judge to be switched off with RAGAS_JUDGE_ENABLED=0."""
    return os.getenv("RAGAS_JUDGE_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def get_judge_config(
    judge_model: str | None = None,
    embedding_model: str | None = None,
    base_url: str | None = None,
    timeout: int | None = None,
) -> tuple[str, str, str, int]:
    """Resolve judge settings from arguments, then environment, then defaults."""
    return (
        judge_model or os.getenv("RAGAS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        embedding_model or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBEDDING_MODEL),
        base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
        int(timeout or os.getenv("RAGAS_JUDGE_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)),
    )


@lru_cache(maxsize=8)
def _build_judge_models(
    judge_model: str,
    embedding_model: str,
    base_url: str,
    timeout: int,
) -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper, RunConfig]:
    """Build (and cache) the Ollama judge LLM and embeddings used by RAGAS."""
    run_config = RunConfig(timeout=timeout, max_retries=2, max_wait=15, max_workers=1)
    judge_llm = LangchainLLMWrapper(
        ChatOllama(
            model=judge_model,
            base_url=base_url,
            temperature=0,
            format="json",
        ),
        run_config=run_config,
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=embedding_model, base_url=base_url),
        run_config=run_config,
    )
    return judge_llm, judge_embeddings, run_config


@lru_cache(maxsize=8)
def _build_metrics(
    judge_model: str,
    embedding_model: str,
    base_url: str,
    timeout: int,
    with_reference: bool,
) -> dict[str, Any]:
    """Instantiate the four RAGAS metrics against the Ollama judge."""
    judge_llm, judge_embeddings, run_config = _build_judge_models(
        judge_model,
        embedding_model,
        base_url,
        timeout,
    )
    context_precision = (
        LLMContextPrecisionWithReference(llm=judge_llm)
        if with_reference
        else LLMContextPrecisionWithoutReference(llm=judge_llm)
    )
    metrics: dict[str, Any] = {
        "faithfulness": Faithfulness(llm=judge_llm),
        "answer_relevancy": ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        "context_precision": context_precision,
        "context_recall": LLMContextRecall(llm=judge_llm),
    }
    for metric in metrics.values():
        metric.init(run_config)
    return metrics


def _run_coroutine(coroutine, timeout: int) -> Any:
    """Run a coroutine from sync code, with or without a live event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    # Already inside an event loop (notebook/async server): use a private loop.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result(timeout=timeout + 30)


async def _score_metric(name: str, metric: Any, sample: SingleTurnSample, timeout: int):
    try:
        return await asyncio.wait_for(metric.single_turn_ascore(sample), timeout=timeout)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "RAGAS metric %s failed: %s",
            name,
            str(exc) or exc.__class__.__name__,
        )
        return exc


async def _score_sample(metrics: dict[str, Any], sample: SingleTurnSample, timeout: int):
    outcomes: dict[str, Any] = {}
    for name, metric in metrics.items():
        outcomes[name] = await _score_metric(name, metric, sample, timeout)
    return outcomes


def _exception_message(error: BaseException) -> str:
    return str(error) or error.__class__.__name__


def _normalise_contexts(contexts: Iterable[Any]) -> list[str]:
    texts: list[str] = []
    for item in contexts:
        text = getattr(item, "page_content", item)
        text = str(text).strip()
        if text:
            texts.append(text)
    return texts


def _clean_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score):
        return None
    return round(max(0.0, min(1.0, score)), 4)


def score_answer(
    question: str,
    answer: str,
    contexts: Sequence[Any],
    reference: str | None = None,
    judge_model: str | None = None,
    embedding_model: str | None = None,
    base_url: str | None = None,
    timeout: int | None = None,
) -> RagasScores:
    """Score one answer with RAGAS faithfulness, relevancy, precision and recall.

    ``reference`` is the ground-truth answer. Context precision and context recall
    need one, so when it is missing the generated answer is used as a proxy and the
    result is reported as ``self-referenced``.
    """
    judge_model, embedding_model, base_url, timeout = get_judge_config(
        judge_model,
        embedding_model,
        base_url,
        timeout,
    )
    context_texts = _normalise_contexts(contexts)
    if not question.strip() or not answer.strip():
        return RagasScores(
            {name: None for name in METRIC_ORDER},
            "unavailable",
            {"input": "Question and answer are required."},
        )
    if not context_texts:
        return RagasScores(
            {name: None for name in METRIC_ORDER},
            "unavailable",
            {"input": "No retrieved context to judge."},
        )

    with_reference = bool(reference and reference.strip())
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=context_texts,
        reference=reference.strip() if with_reference else answer.strip(),
    )
    metrics = _build_metrics(
        judge_model,
        embedding_model,
        base_url,
        timeout,
        with_reference,
    )

    try:
        total_timeout = timeout * len(METRIC_ORDER) + 30
        outcomes = _run_coroutine(_score_sample(metrics, sample, timeout), total_timeout)
    except Exception as exc:  # judge failure must never break the answer
        message = _exception_message(exc)
        logging.getLogger(__name__).warning("RAGAS evaluation failed: %s", message)
        return RagasScores(
            {name: None for name in METRIC_ORDER},
            "unavailable",
            {"judge": message},
        )

    scores: dict[str, float | None] = {}
    errors: dict[str, str] = {}
    for name in METRIC_ORDER:
        outcome = outcomes.get(name)
        if isinstance(outcome, BaseException):
            scores[name] = None
            errors[name] = _exception_message(outcome)
            continue
        scores[name] = _clean_score(outcome)
        if scores[name] is None:
            errors[name] = "The judge did not return a usable score."

    return RagasScores(
        scores,
        "ground-truth" if with_reference else "self-referenced",
        errors,
    )


def format_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def format_scores(scores: RagasScores) -> str:
    """Render the four metrics as an indented block for console output."""
    lines = [f"RAGAS LLM-as-judge scores (reference: {scores.reference_mode})"]
    for name in METRIC_ORDER:
        lines.append(f"  {METRIC_LABELS[name]}: {format_score(scores.scores.get(name))}")
    for name, message in scores.errors.items():
        label = METRIC_LABELS.get(name, name.title())
        lines.append(f"  ! {label}: {message}")
    return "\n".join(lines)


def print_scores(scores: RagasScores) -> None:
    message = format_scores(scores)
    logging.getLogger(__name__).info(message.replace("\n", " | "))
    print(message, flush=True)
