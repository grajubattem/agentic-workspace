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
from ragpipeline.rag_pipeline import build_rag_chain


st.set_page_config(
    page_title="Document RAG",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_rag_chain(source_keys: tuple[str, ...], collection_name: str):
    source_paths = [Path(key) for key in source_keys]
    return build_rag_chain(source_paths, collection_name=collection_name)


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


def main() -> None:
    st.title("Chat With Your Documents")
    st.caption("Upload documents, index them in Chroma, and ask questions through the RAG pipeline.")

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

    source_paths, collection_name = save_uploaded_files(uploaded_files)
    if not source_paths:
        st.info("Upload at least one PDF or Excel file to begin.")
        return

    source_keys = tuple(sorted(str(path.resolve()) for path in source_paths))
    embedding_model, embedding_dimension = get_embedding_model_config()

    try:
        with st.spinner("Preparing the document index..."):
            rag_chain, vector_store = get_rag_chain(source_keys, collection_name)
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

    if st.button("Generate answer", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question first.")
            return

        try:
            with st.spinner("Retrieving context and generating an answer..."):
                answer = rag_chain.invoke(question)
            st.subheader("Answer")
            st.write(answer)
        except Exception as exc:
            st.error(f"Answer generation failed: {exc}")
            return

        with st.expander("Retrieved sources"):
            documents = vector_store.similarity_search(question, k=4)
            for index, document in enumerate(documents, start=1):
                source = document.metadata.get("source", "Unknown source")
                st.markdown(f"**{index}. {source}**")
                st.write(document.page_content)


if __name__ == "__main__":
    main()
