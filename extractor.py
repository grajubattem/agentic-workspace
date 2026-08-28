import json
import os
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from models import InvoiceData


def load_prompts() -> tuple[str, str]:
    prompt_dir = Path(__file__).parent
    text = (prompt_dir / "prompts.txt").read_text(encoding="utf-8")
    guardrails = (prompt_dir / "guardrails.txt").read_text(encoding="utf-8")
    system, user = text.split("\nUSER_PROMPT:\n", 1)
    return system.removeprefix("SYSTEM_PROMPT:\n") + "\n\n" + guardrails, user


def extract_json(response: str) -> dict:
    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Ollama did not return a JSON object")
    return json.loads(response[start : end + 1])


def normalize_null_values(value: object) -> object:
    if isinstance(value, dict):
        return {key: normalize_null_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_null_values(item) for item in value]
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "n/a", "na"}:
        return None
    return value


def normalize_invoice_date(extracted: dict) -> None:
    date_value = extracted.get("invoice_date")
    if not isinstance(date_value, str):
        return
    date_value = date_value.strip()
    for date_format in ("%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            extracted["invoice_date"] = datetime.strptime(date_value, date_format).strftime("%Y-%m-%d")
            return
        except ValueError:
            continue
    extracted["invoice_date"] = None


def extract_invoice(document: str, image_bytes: bytes | None = None) -> dict:
    system_prompt, user_template = load_prompts()
    default_model = "gemma3:4b"
    model_setting = "OLLAMA_VISION_MODEL" if image_bytes else "OLLAMA_TEXT_MODEL"
    model = ChatOllama(
        model=os.getenv(model_setting, os.getenv("OLLAMA_MODEL", default_model)),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
        format="json",
    )
    content: list[dict] = [{"type": "text", "text": user_template.format(document=document)}]
    if image_bytes:
        import base64

        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()}})
    response = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=content)])
    extracted = normalize_null_values(extract_json(response.content))
    if isinstance(extracted, dict):
        normalize_invoice_date(extracted)
    return InvoiceData.model_validate(extracted).model_dump()