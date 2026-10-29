"""
Pydantic schemas for structured extraction.

These schemas define the structure for extracting data from documents
like invoices and contracts using LLM-based extraction.
"""
from pydantic import BaseModel, Field


class Product(BaseModel):
    """Product line item in an invoice."""

    denumire: str = Field(description="Product name/description")
    cantitate: int = Field(description="Quantity")
    pret_unitar: float = Field(description="Unit price")
    total: float = Field(description="Line total")


class Invoice(BaseModel):
    """Extracted invoice data."""

    numar: str = Field(description="Invoice number")
    data: str = Field(description="Invoice date")
    furnizor: str = Field(description="Supplier/vendor name")
    client: str = Field(description="Client/customer name")
    produse: list[Product] = Field(description="List of products/line items")
    subtotal: float = Field(description="Subtotal before tax")
    tva: float = Field(description="VAT/tax amount")
    total: float = Field(description="Total amount including tax")


class Contract(BaseModel):
    """Extracted contract data."""

    numar: str = Field(description="Contract number")
    data_incheiere: str = Field(description="Contract signing date")
    prestator: str = Field(description="Service provider name")
    beneficiar: str = Field(description="Beneficiary/client name")
    valoare: float = Field(description="Contract value")
    durata_luni: int = Field(description="Contract duration in months")
    obligatii_prestator: list[str] = Field(description="Provider obligations")


# Schema registry for dynamic lookup
SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "invoice": Invoice,
    "contract": Contract,
}


def get_schema(schema_type: str) -> type[BaseModel]:
    """Get schema class by type name."""
    if schema_type not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown schema type: {schema_type}. Available: {list(SCHEMA_REGISTRY.keys())}")
    return SCHEMA_REGISTRY[schema_type]
