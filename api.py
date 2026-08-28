import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from extractor import extract_invoice

app = FastAPI(title="Invoice Extractor", version="1.0.0", docs_url=None, redoc_url=None, openapi_url=None)
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


def read_upload(filename: str, content_type: str | None, data: bytes) -> tuple[str, bytes | None]:
    suffix = Path(filename).suffix.lower()
    if content_type in IMAGE_TYPES or suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return "", data
    if suffix == ".pdf" or content_type == "application/pdf":
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
        return text, None
    if content_type == "text/plain" or suffix in {".txt", ".csv"}:
        return data.decode("utf-8", errors="replace"), None
    raise ValueError("Upload an image, PDF, or text file")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract")
async def extract(file: UploadFile = File(...)) -> dict:
    try:
        document, image = read_upload(file.filename or "upload", file.content_type, await file.read())
        return extract_invoice(document, image)
    except Exception as error:
        raise HTTPException(status_code=422, detail=str(error)) from error