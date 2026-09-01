from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
import psycopg
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from psycopg import sql

DEFAULT_FILES = [
    "telangana_daily_temperature_sample_last_1_year.pdf",
    "telangana_daily_vehicle_sales_sample_last_1_year.xlsx",
]

DEFAULT_COLLECTION = "classic_rag_docs"


def get_default_source_files(base_dir: str | Path | None = None) -> List[Path]:
    folder = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    files: List[Path] = []
    for file_name in DEFAULT_FILES:
        path = folder / file_name
        if path.exists():
            files.append(path)
    return files


def build_pgvector_connection_string() -> str:
    return os.getenv(
        "PGVECTOR_CONNECTION_STRING",
        "postgresql+psycopg://postgres:postgres@localhost:5432/postgres",
    )


def ensure_pgvector_extension() -> None:
    conn = psycopg.connect(
        "postgresql://postgres:postgres@localhost:5432/postgres",
        autocommit=True,
    )
    with conn.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.close()


def ensure_vector_table(collection_name: str = DEFAULT_COLLECTION) -> str:
    ensure_pgvector_extension()
    table_name = f"{collection_name.replace('-', '_')}_embedding"
    conn = psycopg.connect(
        "postgresql://postgres:postgres@localhost:5432/postgres",
        autocommit=True,
    )
    with conn.cursor() as cursor:
        cursor.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table} (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    metadata JSONB,
                    embedding vector(768)
                );
                """
            ).format(table=sql.Identifier(table_name))
        )
    conn.close()
    return table_name


def read_pdf_documents(path: Path) -> List[str]:
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = " ".join(text.strip().split())
        if cleaned:
            parts.append(f"[Source: {path.name} | Page {page_number}]\n{cleaned}")
    return parts


def read_excel_documents(path: Path) -> List[str]:
    workbook = pd.ExcelFile(path)
    parts: List[str] = []
    for sheet_name in workbook.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet_name)
        if df.empty:
            continue
        try:
            table_markdown = df.to_markdown(index=False)
        except ImportError:
            table_markdown = df.to_string(index=False)
        table_markdown = table_markdown.strip()
        if table_markdown:
            parts.append(f"[Source: {path.name} | Sheet: {sheet_name}]\n{table_markdown}")
    return parts


def extract_documents_from_paths(paths: Sequence[str | Path]) -> List[Document]:
    documents: List[Document] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text_blocks = read_pdf_documents(path)
        elif suffix in {".xlsx", ".xls"}:
            text_blocks = read_excel_documents(path)
        else:
            continue

        for text_block in text_blocks:
            documents.append(
                Document(
                    page_content=text_block,
                    metadata={"source": path.name, "file_type": suffix.lstrip(".")},
                )
            )
    return documents


def split_documents(documents: Iterable[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(documents))


def create_pgvector_store(source_paths: Sequence[str | Path], collection_name: str = DEFAULT_COLLECTION):
    if not source_paths:
        raise ValueError("No document files were provided for indexing.")

    docs = extract_documents_from_paths(source_paths)
    if not docs:
        raise ValueError("No readable text was found in the uploaded files.")

    ensure_vector_table(collection_name)
    chunks = split_documents(docs)
    embeddings = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    return PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        connection=build_pgvector_connection_string(),
    )


def get_existing_pgvector_store(collection_name: str = DEFAULT_COLLECTION):
    embeddings = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    return PGVector(
        collection_name=collection_name,
        connection=build_pgvector_connection_string(),
        embedding_function=embeddings,
    )
