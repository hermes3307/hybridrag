# Batch Receipt Processing Feature

This OCR receipt processing application now supports batch processing of multiple receipts with automatic aggregation and calculation of totals.

## Features

### Single Receipt Processing
- `/api/v1/process-receipt` - Process a single receipt with OCR and optional LLM parsing
- `/api/v1/process-receipt-simple` - Quick OCR without LLM parsing

### Batch Receipt Processing
- `/api/v1/process-receipts-batch` - Process multiple receipts and get aggregated totals

## Batch Processing Capabilities

The batch processing endpoint processes multiple receipt images/PDFs and returns:

1. **Individual Receipt Results**
   - OCR text extraction
   - LLM-parsed structured data
   - Item lists with prices
   - Merchant information
   - Calculation validation

2. **Aggregated Totals**
   - Grand total across all receipts
   - Total tax, tips, and discounts
   - Item aggregation by name (with quantity and amount)
   - List of unique merchants
   - Success/failure statistics

## API Usage

### Endpoint
```
POST /api/v1/process-receipts-batch
```

### Parameters
- `files` (required): Multiple file uploads (receipt images or PDFs)
- `ocr_engine` (optional): "tesseract", "easyocr", or "paddleocr" (default: paddleocr)
- `preprocess` (optional): true/false (default: true)
- `use_llm` (optional): true/false (default: true)

### Example Using cURL

```bash
curl -X POST "http://localhost:8000/api/v1/process-receipts-batch" \
  -F "files=@receipt1.jpg" \
  -F "files=@receipt2.jpg" \
  -F "files=@receipt3.jpg" \
  -F "ocr_engine=paddleocr" \
  -F "preprocess=true" \
  -F "use_llm=true"
```

### Example Using Python Test Script

```bash
# Install requests if needed
pip install requests

# Run the test script with multiple receipts
python test_batch.py receipts/receipt1.jpg receipts/receipt2.jpg receipts/receipt3.jpg
```

### Example Using Python Requests

```python
import requests

files = [
    ('files', ('receipt1.jpg', open('receipt1.jpg', 'rb'), 'image/jpeg')),
    ('files', ('receipt2.jpg', open('receipt2.jpg', 'rb'), 'image/jpeg')),
    ('files', ('receipt3.jpg', open('receipt3.jpg', 'rb'), 'image/jpeg'))
]

data = {
    'ocr_engine': 'paddleocr',
    'preprocess': 'true',
    'use_llm': 'true'
}

response = requests.post(
    'http://localhost:8000/api/v1/process-receipts-batch',
    files=files,
    data=data
)

result = response.json()
print(f"Grand Total: ${result['aggregated_totals']['grand_total']:.2f}")
```

## Response Format

```json
{
  "total_processing_time": 15.234,
  "ocr_engine_used": "paddleocr",
  "individual_results": [
    {
      "filename": "receipt1.jpg",
      "success": true,
      "raw_text": "...",
      "receipt_summary": {
        "merchant_name": "Store Name",
        "total": 45.67,
        "items": [...],
        ...
      }
    },
    ...
  ],
  "aggregated_totals": {
    "total_receipts": 3,
    "successful_receipts": 3,
    "failed_receipts": 0,
    "grand_total": 123.45,
    "total_tax": 10.50,
    "total_tip": 15.00,
    "total_discount": 5.00,
    "total_items_count": 25,
    "items_by_name": [
      {
        "name": "Coffee",
        "total_quantity": 5.0,
        "total_amount": 25.00,
        "occurrences": 3,
        "category": "Beverages"
      },
      ...
    ],
    "merchants": ["Store 1", "Store 2", "Store 3"],
    "currency": "USD"
  }
}
```

## Key Features

### Automatic Item Aggregation
- Items with the same name (case-insensitive) are automatically combined
- Tracks total quantity and total amount spent per item
- Shows how many receipts contain each item
- Items sorted by total amount (highest to lowest)

### Smart Calculation
- Validates each receipt's totals
- Calculates grand total across all receipts
- Sums up tax, tips, and discounts
- Provides calculation validation results

### Robust Error Handling
- Continues processing even if individual receipts fail
- Provides detailed error messages for failed receipts
- Returns statistics on success/failure rates

## Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation where you can:
- Test endpoints directly from the browser
- See detailed request/response schemas
- Try different parameters

## Use Cases

1. **Expense Tracking**: Process all your receipts from a business trip
2. **Budget Analysis**: Calculate total spending by category
3. **Vendor Analysis**: See spending per merchant
4. **Inventory Management**: Track purchased items and quantities
5. **Financial Reporting**: Generate spending summaries

## Performance Considerations

- First request may be slower due to LLM model loading (lazy initialization)
- Processing time scales linearly with number of receipts
- LLM parsing is optional - disable with `use_llm=false` for faster processing
- Consider processing in batches of 10-20 receipts for optimal performance

## Running the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Start the API server
python api.py

# Server will start on http://0.0.0.0:8000
```

## Requirements

See `requirements.txt` for all dependencies. Main components:
- FastAPI for API framework
- PaddleOCR/EasyOCR/Tesseract for OCR
- Transformers + PyTorch for LLM-based parsing
- Qwen2.5-7B-Instruct model for receipt understanding
