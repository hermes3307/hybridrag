#!/usr/bin/env python3
"""
Test script for OCR receipt processing.
Can be run standalone without the API server.
"""

import sys
import logging
from pathlib import Path

from ocr_engine import OCRService
from llm_parser import Qwen3ReceiptParser
from calculator import ReceiptCalculator
from utils import FileHandler
from models import OCREngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_ocr_only(image_path: str, engine: OCREngine = OCREngine.PADDLEOCR):
    """Test OCR extraction without LLM."""
    logger.info(f"Testing OCR on: {image_path}")
    logger.info(f"Using engine: {engine.value}")

    # Initialize OCR service
    ocr_service = OCRService()

    # Load image
    images = FileHandler.load_image(image_path)
    image = images[0] if isinstance(images, list) else images

    # Extract text
    text = ocr_service.extract_text(image, engine=engine, preprocess=True)

    print("\n" + "=" * 50)
    print("RAW OCR TEXT")
    print("=" * 50)
    print(text)
    print("=" * 50)

    return text


def test_full_pipeline(image_path: str, engine: OCREngine = OCREngine.PADDLEOCR):
    """Test full pipeline including LLM parsing."""
    logger.info(f"Testing full pipeline on: {image_path}")

    # Initialize services
    ocr_service = OCRService()
    llm_parser = Qwen3ReceiptParser(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        use_4bit=True
    )
    calculator = ReceiptCalculator()

    # Load image
    logger.info("Loading image...")
    images = FileHandler.load_image(image_path)
    image = images[0] if isinstance(images, list) else images

    # Extract text
    logger.info("Extracting text with OCR...")
    text = ocr_service.extract_text(image, engine=engine, preprocess=True)

    print("\n" + "=" * 50)
    print("RAW OCR TEXT")
    print("=" * 50)
    print(text)
    print("=" * 50 + "\n")

    # Parse with LLM
    logger.info("Parsing with LLM...")
    receipt = llm_parser.parse_receipt(text)

    if receipt:
        # Validate calculations
        receipt = calculator.validate_and_calculate(receipt)

        # Print summary
        summary = calculator.generate_summary_text(receipt)
        print(summary)

        # Print JSON
        print("\n" + "=" * 50)
        print("JSON OUTPUT")
        print("=" * 50)
        print(receipt.model_dump_json(indent=2))
        print("=" * 50)

    else:
        logger.error("Failed to parse receipt")


def main():
    """Main test function."""
    if len(sys.argv) < 2:
        print("Usage: python test_receipt.py <image_path> [ocr_engine] [mode]")
        print("  image_path: Path to receipt image or PDF")
        print("  ocr_engine: tesseract, easyocr, or paddleocr (default: paddleocr)")
        print("  mode: ocr or full (default: full)")
        print("\nExample:")
        print("  python test_receipt.py receipt.jpg")
        print("  python test_receipt.py receipt.pdf paddleocr full")
        print("  python test_receipt.py receipt.jpg easyocr ocr")
        sys.exit(1)

    image_path = sys.argv[1]

    # Check if file exists
    if not Path(image_path).exists():
        logger.error(f"File not found: {image_path}")
        sys.exit(1)

    # Get OCR engine
    engine_name = sys.argv[2] if len(sys.argv) > 2 else "paddleocr"
    try:
        engine = OCREngine(engine_name.lower())
    except ValueError:
        logger.error(f"Invalid OCR engine: {engine_name}")
        logger.error(f"Valid options: {[e.value for e in OCREngine]}")
        sys.exit(1)

    # Get mode
    mode = sys.argv[3] if len(sys.argv) > 3 else "full"

    # Run test
    if mode == "ocr":
        test_ocr_only(image_path, engine)
    elif mode == "full":
        test_full_pipeline(image_path, engine)
    else:
        logger.error(f"Invalid mode: {mode}. Use 'ocr' or 'full'")
        sys.exit(1)


if __name__ == "__main__":
    main()
