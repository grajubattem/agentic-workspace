from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InvoiceData(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    invoice_number: str | None = None
    invoice_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    vendor: str | None = None
    bill_to: str | None = None
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    payment_terms: str | None = None
    currency: str | None = None
    confidence: dict[str, Any] = Field(default_factory=dict, alias="_confidence")