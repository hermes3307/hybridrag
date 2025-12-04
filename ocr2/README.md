# OCR Receipt Processing System

A comprehensive OCR-based receipt processing system with LLM parsing, calculation validation, and multi-format support.

## Features

- **Multi-OCR Engine Support**: Choose between Tesseract, EasyOCR, or PaddleOCR
- **Advanced Preprocessing**: Image enhancement for better OCR accuracy
- **LLM-Powered Parsing**: Uses Qwen3 (6B/8B) for intelligent receipt data extraction
- **Automatic Calculation**: Validates receipt totals and detects discrepancies
- **Multi-Format Support**: Process PDF, JPG, PNG, and other image formats
- **REST API**: FastAPI-based endpoints for easy integration
- **Lightweight Deployment**: 4-bit quantization for resource-efficient LLM inference

## Architecture

```
┌─────────────────┐
│  Receipt File   │
│ (PDF/JPG/PNG)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessor   │ ─── Image enhancement
│  (OpenCV)       │ ─── Denoising, deskewing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   OCR Engine    │ ─── Tesseract
│                 │ ─── EasyOCR
│                 │ ─── PaddleOCR
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Raw Text       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Parser     │ ─── Qwen3 6B/8B
│  (Qwen3)        │ ─── Structured extraction
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Calculator     │ ─── Validation
│  & Validator    │ ─── Summary generation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Receipt Data   │
│  (JSON)         │
└─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster processing)
- Tesseract OCR (optional, if using Tesseract engine)

### Setup

1. **Clone and navigate to the project:**
```bash
cd /home/pi/hybridrag/ocr2
```

2. **Install system dependencies (for Tesseract):**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download Qwen3 model (first run will download automatically):**
The model will be downloaded from HuggingFace on first use (~4GB for 7B model with 4-bit quantization).

## Usage

### Option 1: REST API

Start the API server:

```bash
python api.py
```

The API will be available at `http://localhost:8000`

**API Endpoints:**

1. **Process Receipt (Full Pipeline):**
```bash
curl -X POST "http://localhost:8000/api/v1/process-receipt" \
  -F "file=@receipt.jpg" \
  -F "ocr_engine=paddleocr" \
  -F "preprocess=true" \
  -F "use_llm=true"
```

2. **Process Receipt (OCR Only):**
```bash
curl -X POST "http://localhost:8000/api/v1/process-receipt-simple" \
  -F "file=@receipt.pdf" \
  -F "ocr_engine=easyocr"
```

3. **Health Check:**
```bash
curl http://localhost:8000/health
```

### Option 2: Command Line

Test with a single receipt:

```bash
# Full pipeline (OCR + LLM parsing)
python test_receipt.py receipt.jpg

# OCR only (no LLM)
python test_receipt.py receipt.jpg paddleocr ocr

# With different OCR engines
python test_receipt.py receipt.pdf easyocr full
python test_receipt.py receipt.png tesseract full
```

### Option 3: Python Code

```python
from ocr_engine import OCRService
from llm_parser import Qwen3ReceiptParser
from calculator import ReceiptCalculator
from utils import FileHandler
from models import OCREngine

# Initialize services
ocr_service = OCRService()
llm_parser = Qwen3ReceiptParser(use_4bit=True)
calculator = ReceiptCalculator()

# Load and process image
image = FileHandler.load_image("receipt.jpg")
if isinstance(image, list):
    image = image[0]

# Extract text
text = ocr_service.extract_text(
    image,
    engine=OCREngine.PADDLEOCR,
    preprocess=True
)

# Parse with LLM
receipt = llm_parser.parse_receipt(text)

# Validate calculations
receipt = calculator.validate_and_calculate(receipt)

# Generate summary
summary = calculator.generate_summary_text(receipt)
print(summary)
```

## Configuration

Edit `config.py` or use environment variables:

```bash
# LLM Model Selection
export LLM_MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"  # Lighter model

# API Settings
export API_HOST="0.0.0.0"
export API_PORT="8000"

# Processing Settings
export PDF_DPI="300"
export USE_4BIT_QUANTIZATION="true"
```

## OCR Engine Comparison

| Engine      | Speed | Accuracy | GPU Support | Memory Usage |
|-------------|-------|----------|-------------|--------------|
| Tesseract   | Fast  | Good     | No          | Low          |
| EasyOCR     | Medium| Better   | Yes         | Medium       |
| PaddleOCR   | Fast  | Best     | Yes         | Low          |

**Recommendation:** PaddleOCR for best balance of speed and accuracy.

## LLM Model Options

| Model                     | Size  | Speed | Accuracy | Memory (4-bit) |
|---------------------------|-------|-------|----------|----------------|
| Qwen2.5-1.5B-Instruct    | 1.5B  | Fast  | Good     | ~2 GB          |
| Qwen2.5-3B-Instruct      | 3B    | Fast  | Better   | ~3 GB          |
| Qwen2.5-7B-Instruct      | 7B    | Medium| Best     | ~5 GB          |
| Qwen2.5-14B-Instruct     | 14B   | Slow  | Excellent| ~10 GB         |

**Recommendation:** Qwen2.5-7B-Instruct for good balance (default).

## Response Format

```json
{
  "success": true,
  "raw_text": "STORE NAME\n123 Main St\n...",
  "receipt_summary": {
    "merchant_name": "STORE NAME",
    "merchant_address": "123 Main St",
    "date": "2024-01-15",
    "time": "14:30",
    "items": [
      {
        "name": "Product A",
        "quantity": 2.0,
        "unit_price": 10.50,
        "total_price": 21.00,
        "category": "Electronics"
      }
    ],
    "subtotal": 21.00,
    "tax": 1.68,
    "total": 22.68,
    "calculated_total": 22.68,
    "calculation_matches": true
  },
  "processing_time": 3.45,
  "ocr_engine_used": "paddleocr"
}
```

## Performance Tips

1. **Use PaddleOCR** for best results on receipts
2. **Enable preprocessing** for low-quality images
3. **Use 4-bit quantization** to reduce memory usage
4. **Use smaller LLM** (3B) if speed is critical
5. **Process PDFs at 300 DPI** for optimal quality
6. **Use GPU** if available for 2-3x speedup

## Troubleshooting

### OCR returns empty text
- Ensure image is readable (not too blurry or dark)
- Try different OCR engines
- Enable preprocessing
- Increase PDF DPI to 400-600

### LLM parsing fails
- Check raw OCR text quality
- Try a larger model (7B instead of 3B)
- Ensure sufficient GPU/RAM memory
- Check logs for specific errors

### Out of memory errors
- Enable 4-bit quantization
- Use smaller model (3B or 1.5B)
- Reduce batch size
- Close other applications

### API is slow
- Use GPU if available
- Reduce image resolution
- Disable LLM parsing for faster processing
- Use OCR-only endpoint

## Project Structure

```
ocr2/
├── api.py                 # FastAPI endpoints
├── calculator.py          # Receipt calculation & validation
├── config.py             # Configuration settings
├── llm_parser.py         # Qwen3 LLM integration
├── models.py             # Pydantic data models
├── ocr_engine.py         # OCR service (multi-engine)
├── preprocessor.py       # Image preprocessing
├── utils.py              # File handling utilities
├── test_receipt.py       # Test script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Support

For issues and questions, please create an issue in the repository.
