from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama

from embedding_pipeline import (
    DEFAULT_COLLECTION,
    create_pgvector_store,
)


def build_rag_chain(
    source_paths: Sequence[str | Path],
    llm_model: str | None = None,
    embedding_model: str | None = None,
    base_url: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
):
    vector_store = create_pgvector_store(
        source_paths,
        collection_name=collection_name,
    )
    llm = ChatOllama(
        model=llm_model or os.getenv("OLLAMA_LLM_MODEL", "llama3.2"),
        base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the user's question using only the context below.
        If the answer is not mentioned in the context, say that it is not available in the provided files.
        Include citations using the source file names in square brackets, for example [report.pdf].

        Context:
        {context}

        Question: {question}
        """
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    rag_chain = (
        RunnableParallel(
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
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
