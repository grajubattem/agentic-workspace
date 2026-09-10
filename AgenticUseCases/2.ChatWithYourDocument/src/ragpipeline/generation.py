from __future__ import annotations

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama


def build_generation_chain(
    llm_model: str | None = None,
    base_url: str | None = None,
) -> Runnable:
    """Build the answer-generation chain from retrieved context."""
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
    return prompt | llm | StrOutputParser()


def generate_answer(
    question: str,
    context: list,
    llm_model: str | None = None,
    base_url: str | None = None,
) -> str:
    """Generate an answer from a question and retrieved context."""
    if not question or not question.strip():
        raise ValueError("Please provide a valid question.")

    return build_generation_chain(llm_model=llm_model, base_url=base_url).invoke(
        {"context": context, "question": question}
    )
