from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from embedding_pipeline import DEFAULT_FILES, get_default_source_files
from rag_pipeline import build_rag_chain

st.set_page_config(page_title="Classic RAG", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def get_cached_rag_chain(source_keys: tuple[str, ...]):
    source_files = [Path(key) for key in source_keys]
    return build_rag_chain(source_files)


def save_uploaded_files(uploaded_files):
    if not uploaded_files:
        return get_default_source_files()

    saved_paths: list[Path] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="rag_docs_"))
    for uploaded in uploaded_files:
        if uploaded is None:
            continue
        file_name = uploaded.name
        if not file_name:
            continue
        destination = temp_dir / file_name
        destination.write_bytes(uploaded.getvalue())
        saved_paths.append(destination)
    return saved_paths


def main():
    st.markdown(
        """
        <style>
        .stApp { background: #f3f1f4; }
        .block-container { padding-top: 2rem; }
        div[data-testid="stFileUploader"] { margin-bottom: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Chat With Your Document")
    st.caption("Upload a PDF and/or Excel file, ask a question, and get a grounded answer with citations.")

    with st.sidebar:
        st.subheader("Source documents")
        uploaded_files = st.file_uploader(
            "Upload PDF / Excel files",
            type=["pdf", "xlsx", "xls"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) selected")
        else:
            default_files = get_default_source_files()
            st.caption(f"Using default sample files: {', '.join(f.name for f in default_files)}")

    source_files = save_uploaded_files(uploaded_files)

    if not source_files:
        st.warning("No supported files were found. Please upload a PDF or Excel file.")
        return

    source_keys = tuple(sorted(str(path.resolve()) for path in source_files))
    if "rag_chain" not in st.session_state:
        try:
            with st.spinner("Loading and embedding documents..."):
                rag_chain, _ = get_cached_rag_chain(source_keys)
                st.session_state["rag_chain"] = rag_chain
        except Exception as exc:  # pragma: no cover - UI safety
            st.error(f"Failed to build the document index: {exc}")
            return

    st.write("### Ask a question")
    question = st.text_input(
        "Question",
        placeholder="Example: What is the operating temperature range?",
    )

    if st.button("Generate answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question first.")
            return

        try:
            with st.spinner("Searching the document store and generating an answer..."):
                answer = st.session_state["rag_chain"].invoke(question)
            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources used")
            # The retriever is embedded in the chain, so we show the selected files from the active index.
            sources = []
            for path in get_default_source_files():
                if path.exists():
                    sources.append(path.name)
            for source in sources:
                st.write(f"- {source}")
        except Exception as exc:  # pragma: no cover - UI safety
            st.error(f"Answer generation failed: {exc}")


if __name__ == "__main__":
    main()
