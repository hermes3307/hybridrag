# Quick Start Guide

Get started with OCR Receipt Processing in 5 minutes!

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Install Tesseract for Tesseract OCR engine
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract
```

## Quick Test

### Method 1: Command Line (Fastest to test)

```bash
# Create a test with any receipt image
python test_receipt.py your_receipt.jpg

# Or test OCR only (no LLM, faster)
python test_receipt.py your_receipt.jpg paddleocr ocr
```

### Method 2: API Server

```bash
# 1. Start the API server
python api.py

# 2. In another terminal, test with curl
curl -X POST "http://localhost:8000/api/v1/process-receipt-simple" \
  -F "file=@your_receipt.jpg" \
  -F "ocr_engine=paddleocr"
```

### Method 3: Python Script

Create `quick_test.py`:

```python
from ocr_engine import OCRService
from models import OCREngine

# Initialize
ocr = OCRService()

# Process receipt
text = ocr.extract_text(
    "your_receipt.jpg",
    engine=OCREngine.PADDLEOCR,
    preprocess=True
)

print(text)
```

Run it:
```bash
python quick_test.py
```

## Common Use Cases

### 1. Just Extract Text (No LLM)

```python
from ocr_engine import OCRService
from models import OCREngine

ocr = OCRService()
text = ocr.extract_text("receipt.jpg", OCREngine.PADDLEOCR)
print(text)
```

### 2. Full Pipeline with LLM Parsing

```python
from ocr_engine import OCRService
from llm_parser import Qwen3ReceiptParser
from calculator import ReceiptCalculator
from models import OCREngine

# Initialize
ocr = OCRService()
llm = Qwen3ReceiptParser(use_4bit=True)
calc = ReceiptCalculator()

# Process
text = ocr.extract_text("receipt.jpg", OCREngine.PADDLEOCR)
receipt = llm.parse_receipt(text)
receipt = calc.validate_and_calculate(receipt)

# Print summary
summary = calc.generate_summary_text(receipt)
print(summary)
```

### 3. Process PDF Receipt

```python
from utils import FileHandler
from ocr_engine import OCRService
from models import OCREngine

# Load PDF
images = FileHandler.load_image("receipt.pdf")

# Process first page
ocr = OCRService()
text = ocr.extract_text(images[0], OCREngine.PADDLEOCR)
print(text)
```

## Choosing the Right Configuration

### For Speed (Quick Testing)
```python
# Use lightweight model, OCR only
ocr = OCRService()
text = ocr.extract_text("receipt.jpg", OCREngine.PADDLEOCR, preprocess=False)
```

### For Accuracy
```python
# Use full pipeline with preprocessing
from ocr_engine import OCRService
from llm_parser import Qwen3ReceiptParser
from models import OCREngine

ocr = OCRService()
llm = Qwen3ReceiptParser(
    model_name="Qwen/Qwen2.5-7B-Instruct",  # or 14B for even better
    use_4bit=True
)

text = ocr.extract_text("receipt.jpg", OCREngine.PADDLEOCR, preprocess=True)
receipt = llm.parse_receipt(text)
```

### For Low Memory
```python
# Use smaller model with 4-bit quantization
llm = Qwen3ReceiptParser(
    model_name="Qwen/Qwen2.5-3B-Instruct",  # or 1.5B
    use_4bit=True
)
```

## Troubleshooting

**Problem: OCR returns empty text**
- Solution: Enable preprocessing: `preprocess=True`

**Problem: Out of memory**
- Solution: Use smaller model: `Qwen/Qwen2.5-3B-Instruct`
- Solution: Enable 4-bit: `use_4bit=True`

**Problem: Slow processing**
- Solution: Disable LLM: Use `OCRService` only
- Solution: Use smaller model: `Qwen/Qwen2.5-3B-Instruct`

**Problem: API won't start**
- Solution: Check port 8000 is free: `lsof -i :8000`
- Solution: Change port in `config.py`

## Next Steps

- Read [README.md](README.md) for full documentation
- Check [examples.sh](examples.sh) for API examples
- Customize [config.py](config.py) for your needs
- Try different OCR engines for comparison

## Support

For issues: Create an issue in the repository
For questions: Check README.md troubleshooting section
