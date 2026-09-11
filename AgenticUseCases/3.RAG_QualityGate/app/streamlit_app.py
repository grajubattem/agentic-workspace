from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embeddingpipeline.embedding import get_embedding_model_config
from evaluation.ragas_judge import (
    METRIC_DESCRIPTIONS,
    METRIC_LABELS,
    METRIC_ORDER,
    format_score,
    get_judge_config,
    judge_enabled,
)
from ragpipeline.rag_pipeline import answer_with_quality_gate
from ragpipeline.retrieval import get_vector_store


st.set_page_config(
    page_title="Document RAG",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_indexed_store(source_keys: tuple[str, ...], collection_name: str):
    source_paths = [Path(key) for key in source_keys]
    return get_vector_store(source_paths, collection_name=collection_name)


def save_uploaded_files(uploaded_files) -> tuple[list[Path], str]:
    if not uploaded_files:
        return [], ""

    file_bytes: list[bytes] = []
    saved_paths: list[Path] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="document_rag_"))

    for uploaded_file in uploaded_files:
        content = uploaded_file.getvalue()
        file_bytes.append(content)
        destination = temp_dir / Path(uploaded_file.name).name
        destination.write_bytes(content)
        saved_paths.append(destination)

    digest = hashlib.sha256()
    for content in file_bytes:
        digest.update(content)
    collection_name = f"uploaded_docs_{digest.hexdigest()[:16]}"
    return saved_paths, collection_name


def render_ragas_scores(scores) -> None:
    st.subheader("RAGAS LLM-as-judge scores")
    columns = st.columns(len(METRIC_ORDER))
    for column, metric_name in zip(columns, METRIC_ORDER):
        column.metric(
            METRIC_LABELS[metric_name],
            format_score(scores.scores.get(metric_name)),
            help=METRIC_DESCRIPTIONS[metric_name],
        )

    if scores.reference_mode == "ground-truth":
        st.caption("Context precision and recall were scored against the expected answer you provided.")
    elif scores.reference_mode == "self-referenced":
        st.caption(
            "No expected answer was provided, so the generated answer was used as the reference "
            "for context precision and recall. Add an expected answer for stricter scoring."
        )
    elif scores.reference_mode == "disabled":
        st.caption("The RAGAS judge is turned off. Enable it in the sidebar to score this answer.")

    for metric_name, message in scores.errors.items():
        label = METRIC_LABELS.get(metric_name, metric_name.replace("_", " ").title())
        st.warning(f"{label}: {message}")


def main() -> None:
    st.title("Chat With Your Documents")
    st.caption("Upload documents, index them in Chroma, and ask questions through the RAG pipeline.")

    default_judge_model, _, base_url, timeout = get_judge_config()

    with st.sidebar:
        st.subheader("Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF or Excel files",
            type=["pdf", "xlsx", "xls"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            st.write(f"{len(uploaded_files)} file(s) selected")
            for uploaded_file in uploaded_files:
                st.caption(uploaded_file.name)

        st.subheader("Quality gate")
        run_ragas = st.toggle(
            "Run RAGAS LLM-as-judge",
            value=judge_enabled(),
            help="Scores faithfulness, answer relevancy, context precision and context recall. Adds latency.",
        )
        judge_model = st.text_input("Judge model", value=default_judge_model)
        st.caption(f"Judge endpoint: {base_url}")
        st.caption(f"Judge timeout: {timeout}s")

    source_paths, collection_name = save_uploaded_files(uploaded_files)
    if not source_paths:
        st.info("Upload at least one PDF or Excel file to begin.")
        return

    source_keys = tuple(sorted(str(path.resolve()) for path in source_paths))
    embedding_model, embedding_dimension = get_embedding_model_config()

    try:
        with st.spinner("Preparing the document index..."):
            vector_store = get_indexed_store(source_keys, collection_name)
    except Exception as exc:
        st.error(f"Failed to build the document index: {exc}")
        st.stop()

    st.sidebar.success("Document index ready")
    st.sidebar.caption(f"Embedding model: {embedding_model}")
    st.sidebar.caption(f"Vector dimension: {embedding_dimension}")
    st.sidebar.caption(f"Stored chunks: {vector_store._collection.count()}")

    question = st.text_area(
        "Ask a question",
        placeholder="Example: Which car has the highest price?",
        height=100,
    )
    reference = st.text_area(
        "Expected answer (optional)",
        placeholder="Provide the ground-truth answer to score context precision and recall against it.",
        height=68,
    )

    if st.button("Generate answer", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question first.")
            return

        try:
            spinner_text = (
                "Retrieving context, generating an answer and running the RAGAS judge..."
                if run_ragas
                else "Retrieving context and generating an answer..."
            )
            with st.spinner(spinner_text):
                response = answer_with_quality_gate(
                    question,
                    [Path(key) for key in source_keys],
                    collection_name=collection_name,
                    reference=reference or None,
                    run_ragas=run_ragas,
                    judge_model=judge_model or None,
                )
        except Exception as exc:
            st.error(f"Answer generation failed: {exc}")
            return

        st.subheader("Answer")
        st.write(response.answer)

        render_ragas_scores(response.ragas)

        if response.evaluation is not None:
            st.caption(
                f"Local checks — retrieval relevance: {response.evaluation.relevance_score:.3f}, "
                f"groundedness: {response.evaluation.groundedness_score:.3f}"
            )

        with st.expander("Retrieved sources"):
            for index, document in enumerate(response.documents, start=1):
                source = document.metadata.get("source", "Unknown source")
                score = (
                    response.relevance_scores[index - 1]
                    if index - 1 < len(response.relevance_scores)
                    else None
                )
                heading = f"**{index}. {source}**"
                if score is not None:
                    heading += f" — relevance {score:.3f}"
                st.markdown(heading)
                st.write(document.page_content)


if __name__ == "__main__":
    main()
