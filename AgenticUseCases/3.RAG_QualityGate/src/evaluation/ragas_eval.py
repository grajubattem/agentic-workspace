from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) in sys.path:
    sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from ragpipeline.generation import build_generation_chain
from ragpipeline.retrieval import retrieve_documents
from evaluation.ragas_judge import METRIC_ORDER, RagasScores, score_answer


METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def load_questions(path: Path) -> list[dict[str, str]]:
    """Load questions and reference answers from a JSON array or JSONL file."""
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(records, list) or not records:
        raise ValueError("The evaluation file must contain a non-empty JSON list.")

    questions: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Evaluation item {index} must be a JSON object.")
        question = str(record.get("question", "")).strip()
        reference = str(record.get("reference", record.get("ground_truth", ""))).strip()
        if not question or not reference:
            raise ValueError(f"Evaluation item {index} needs question and reference fields.")
        questions.append({"question": question, "reference": reference})
    return questions


def build_evaluation_rows(
    questions: list[dict[str, str]],
    source_paths: list[Path],
    collection_name: str,
    llm_model: str,
    top_k: int,
) -> list[dict[str, Any]]:
    generation_chain = build_generation_chain(llm_model=llm_model)
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(questions, start=1):
        print(f"Preparing question {index}/{len(questions)}...", flush=True)
        documents = retrieve_documents(
            item["question"],
            source_paths,
            collection_name=collection_name,
            top_k=top_k,
        )
        answer = generation_chain.invoke(
            {"context": documents, "question": item["question"]}
        )
        rows.append(
            {
                "user_input": item["question"],
                "response": answer,
                "retrieved_contexts": [document.page_content for document in documents],
                "reference": item["reference"],
            }
        )
        print(f"Generated answer for question {index}/{len(questions)}.", flush=True)
    return rows


def evaluate_questions(
    rows: list[dict[str, Any]],
    judge_model: str,
    base_url: str,
) -> list[RagasScores]:
    scores: list[RagasScores] = []
    for index, row in enumerate(rows, start=1):
        print(f"Judging question {index}/{len(rows)}...", flush=True)
        scores.append(
            score_answer(
                row["user_input"],
                row["response"],
                row["retrieved_contexts"],
                reference=row["reference"],
                judge_model=judge_model,
                base_url=base_url,
            )
        )
    return scores


def print_results(rows: list[dict[str, Any]], scores: list[RagasScores]) -> None:
    for index, (row, score) in enumerate(zip(rows, scores), start=1):
        print(f"\nQuestion {index}: {row['user_input']}")
        print(f"  Faithfulness: {score.faithfulness if score.faithfulness is not None else 'n/a'}")
        print(f"  Answer Relevancy: {score.answer_relevancy if score.answer_relevancy is not None else 'n/a'}")
        print(f"  Context Precision: {score.context_precision if score.context_precision is not None else 'n/a'}")
        print(f"  Context Recall: {score.context_recall if score.context_recall is not None else 'n/a'}")
        for name, message in score.errors.items():
            print(f"  ! {name}: {message}")

    print("\nAverage scores:", flush=True)
    for metric_name in METRIC_ORDER:
        label = metric_name.replace("_", " ").title()
        values = [score.scores[metric_name] for score in scores if score.scores[metric_name] is not None]
        average = sum(values) / len(values) if values else None
        print(f"  {label}: {average if average is not None else 'n/a'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a RAG pipeline with RAGAS and an Ollama judge.")
    parser.add_argument("--questions", type=Path, required=True, help="JSON or JSONL file with question/reference pairs.")
    parser.add_argument("--source", type=Path, action="append", required=True, help="Source PDF/XLSX file; repeat for multiple files.")
    parser.add_argument("--collection", default="ragas_eval_docs")
    parser.add_argument("--top-k", type=int, default=int(os.getenv("RAGAS_TOP_K", "2")))
    parser.add_argument("--llm-model", default=os.getenv("OLLAMA_LLM_MODEL", "llama3.2"))
    parser.add_argument("--judge-model", default=os.getenv("RAGAS_JUDGE_MODEL", "llama3.2"))
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions(args.questions)
    rows = build_evaluation_rows(
        questions,
        args.source,
        collection_name=args.collection,
        llm_model=args.llm_model,
        top_k=args.top_k,
    )
    print("Running RAGAS LLM-as-judge evaluation...", flush=True)
    scores = evaluate_questions(rows, args.judge_model, args.base_url)
    print_results(rows, scores)


if __name__ == "__main__":
    main()