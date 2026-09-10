from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Sequence

import pandas as pd
from langchain_core.documents import Document
from pypdf import PdfReader

from .vector_store import create_chroma_store, get_existing_chroma_store

DEFAULT_FILES = [
    "telangana_daily_temperature_sample_last_1_year.pdf",
    "telangana_daily_vehicle_sales_sample_last_1_year.xlsx",
]
DEFAULT_TARGET_PDF = "Certificate14188585.pdf"

DEFAULT_COLLECTION = "classic_rag_docs"
DEFAULT_CHROMA_DIRECTORY = "chroma_db"
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
DEFAULT_EMBEDDING_LENGTH = 1024


def get_default_source_files(base_dir: str | Path | None = None) -> List[Path]:
    folder = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    files: List[Path] = []

    preferred = [folder / DEFAULT_TARGET_PDF, *[folder / file_name for file_name in DEFAULT_FILES]]
    for path in preferred:
        if path.exists():
            files.append(path)

    if files:
        return files

    for pattern in ("*.pdf", "*.xlsx", "*.xls"):
        for path in sorted(folder.glob(pattern)):
            if path.is_file():
                files.append(path)
    return files


def get_embedding_model_config() -> tuple[str, int]:
    model = os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBEDDING_MODEL)
    model_dimensions = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }
    return model, model_dimensions.get(model, DEFAULT_EMBEDDING_LENGTH)


def get_chroma_directory() -> str:
    return os.getenv("CHROMA_PERSIST_DIRECTORY", DEFAULT_CHROMA_DIRECTORY)



def get_collection_embedding_dimension(collection_name: str = DEFAULT_COLLECTION) -> int:
    return get_embedding_model_config()[1]


def collection_exists(collection_name: str = DEFAULT_COLLECTION) -> bool:
    directory = Path(get_chroma_directory())
    if not directory.exists():
        return False

    try:
        store = get_existing_chroma_store(collection_name)
        return store._collection.count() > 0
    except Exception:
        return False


def reset_chroma_collection(collection_name: str = DEFAULT_COLLECTION) -> None:
    store = get_existing_chroma_store(collection_name)
    store.delete_collection()


def get_or_create_chroma_store(
    source_paths: Sequence[str | Path],
    collection_name: str = DEFAULT_COLLECTION,
):
    if collection_exists(collection_name):
        return get_existing_chroma_store(collection_name)
    return create_chroma_store(source_paths, collection_name=collection_name)


def read_pdf_documents(path: Path) -> List[str]:
    warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")
    reader = PdfReader(str(path), strict=False)
    parts: List[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
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


