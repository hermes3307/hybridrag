from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class OCREngine(str, Enum):
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"


class ReceiptItem(BaseModel):
    name: str
    quantity: Optional[float] = 1.0
    unit_price: Optional[float] = None
    total_price: float
    category: Optional[str] = None


class ReceiptSummary(BaseModel):
    merchant_name: Optional[str] = None
    merchant_address: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    items: List[ReceiptItem] = []
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    tip: Optional[float] = None
    discount: Optional[float] = None
    total: float
    currency: str = "USD"
    payment_method: Optional[str] = None

    # Calculation validation
    calculated_subtotal: Optional[float] = None
    calculated_total: Optional[float] = None
    calculation_matches: bool = True
    calculation_difference: Optional[float] = None


class OCRRequest(BaseModel):
    ocr_engine: OCREngine = OCREngine.PADDLEOCR
    preprocess: bool = True
    use_llm: bool = True


class OCRResponse(BaseModel):
    success: bool
    raw_text: str
    receipt_summary: Optional[ReceiptSummary] = None
    processing_time: float
    ocr_engine_used: str
    error: Optional[str] = None


class ItemAggregate(BaseModel):
    """Aggregated data for a specific item across multiple receipts."""
    name: str
    total_quantity: float
    total_amount: float
    occurrences: int  # Number of receipts containing this item
    category: Optional[str] = None


class AggregatedTotals(BaseModel):
    """Aggregated totals across all processed receipts."""
    total_receipts: int
    successful_receipts: int
    failed_receipts: int
    grand_total: float  # Sum of all receipt totals
    total_tax: float
    total_tip: float
    total_discount: float
    total_items_count: int
    items_by_name: List[ItemAggregate] = []
    currency: str = "USD"
    merchants: List[str] = []  # Unique merchants


class BatchReceiptResult(BaseModel):
    """Result for a single receipt in batch processing."""
    filename: str
    success: bool
    raw_text: Optional[str] = None
    receipt_summary: Optional[ReceiptSummary] = None
    error: Optional[str] = None


class BatchReceiptResponse(BaseModel):
    """Response for batch receipt processing."""
    total_processing_time: float
    ocr_engine_used: str
    individual_results: List[BatchReceiptResult]
    aggregated_totals: AggregatedTotals
