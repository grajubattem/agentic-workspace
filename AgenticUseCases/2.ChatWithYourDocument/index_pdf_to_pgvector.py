from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("embedding_pipeline.py")

spec = importlib.util.spec_from_file_location("embedpipe", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)


def main() -> None:
    pdf_path = module.get_default_source_files()[0]
    print(f"PDF path: {pdf_path}")
    print(f"PDF exists: {pdf_path.exists()}")

    docs = module.extract_documents_from_paths([pdf_path])
    print(f"Extracted documents: {len(docs)}")

    chunks = module.split_documents(docs)
    print(f"Split chunks: {len(chunks)}")

    store = module.create_pgvector_store([pdf_path], collection_name="classic_rag_docs")
    print(f"Store type: {type(store).__name__}")
    print(f"Saved {len(chunks)} chunks to pgvector collection 'classic_rag_docs'.")


if __name__ == "__main__":
    main()
