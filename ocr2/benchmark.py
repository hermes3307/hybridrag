#!/usr/bin/env python3
"""
Benchmark script to compare different OCR engines and configurations.
"""

import time
import logging
from pathlib import Path
import sys

from ocr_engine import OCRService
from models import OCREngine
from utils import FileHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_ocr_engine(image_path: str, engine: OCREngine, runs: int = 3):
    """Benchmark a specific OCR engine."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {engine.value.upper()}")
    print(f"{'='*60}")

    ocr_service = OCRService()

    # Load image once
    images = FileHandler.load_image(image_path)
    image = images[0] if isinstance(images, list) else images

    results = []

    for i in range(runs):
        print(f"\nRun {i+1}/{runs}...")

        # Without preprocessing
        start = time.time()
        try:
            text_no_prep = ocr_service.extract_text(
                image.copy(),
                engine=engine,
                preprocess=False
            )
            time_no_prep = time.time() - start
            chars_no_prep = len(text_no_prep)
        except Exception as e:
            logger.error(f"Error without preprocessing: {e}")
            time_no_prep = 0
            chars_no_prep = 0

        # With preprocessing
        start = time.time()
        try:
            text_prep = ocr_service.extract_text(
                image.copy(),
                engine=engine,
                preprocess=True
            )
            time_prep = time.time() - start
            chars_prep = len(text_prep)
        except Exception as e:
            logger.error(f"Error with preprocessing: {e}")
            time_prep = 0
            chars_prep = 0

        results.append({
            'no_prep_time': time_no_prep,
            'no_prep_chars': chars_no_prep,
            'prep_time': time_prep,
            'prep_chars': chars_prep
        })

        print(f"  Without preprocessing: {time_no_prep:.2f}s, {chars_no_prep} chars")
        print(f"  With preprocessing:    {time_prep:.2f}s, {chars_prep} chars")

    # Calculate averages
    avg_no_prep_time = sum(r['no_prep_time'] for r in results) / runs
    avg_prep_time = sum(r['prep_time'] for r in results) / runs
    avg_no_prep_chars = sum(r['no_prep_chars'] for r in results) / runs
    avg_prep_chars = sum(r['prep_chars'] for r in results) / runs

    print(f"\n{'='*60}")
    print(f"AVERAGE RESULTS ({runs} runs)")
    print(f"{'='*60}")
    print(f"Without preprocessing: {avg_no_prep_time:.2f}s, {avg_no_prep_chars:.0f} chars")
    print(f"With preprocessing:    {avg_prep_time:.2f}s, {avg_prep_chars:.0f} chars")
    print(f"{'='*60}")

    return {
        'engine': engine.value,
        'avg_no_prep_time': avg_no_prep_time,
        'avg_prep_time': avg_prep_time,
        'avg_no_prep_chars': avg_no_prep_chars,
        'avg_prep_chars': avg_prep_chars
    }


def main():
    """Main benchmark function."""
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <image_path> [runs]")
        print("  image_path: Path to receipt image")
        print("  runs: Number of runs per test (default: 3)")
        print("\nExample:")
        print("  python benchmark.py receipt.jpg")
        print("  python benchmark.py receipt.jpg 5")
        sys.exit(1)

    image_path = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    if not Path(image_path).exists():
        logger.error(f"File not found: {image_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"OCR ENGINE BENCHMARK")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Runs per engine: {runs}")
    print(f"{'='*60}")

    # Test all engines
    all_results = []
    engines_to_test = [OCREngine.PADDLEOCR, OCREngine.EASYOCR]

    # Try Tesseract if available
    try:
        import pytesseract
        engines_to_test.insert(0, OCREngine.TESSERACT)
    except:
        logger.warning("Tesseract not available, skipping...")

    for engine in engines_to_test:
        try:
            result = benchmark_ocr_engine(image_path, engine, runs)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to benchmark {engine.value}: {e}")

    # Print comparison
    if len(all_results) > 1:
        print(f"\n\n{'='*60}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"\n{'Engine':<15} {'No Prep':<12} {'With Prep':<12} {'Winner'}")
        print(f"{'-'*60}")

        for result in all_results:
            no_prep = f"{result['avg_no_prep_time']:.2f}s"
            prep = f"{result['avg_prep_time']:.2f}s"
            print(f"{result['engine']:<15} {no_prep:<12} {prep:<12}")

        # Find fastest
        fastest_no_prep = min(all_results, key=lambda x: x['avg_no_prep_time'])
        fastest_prep = min(all_results, key=lambda x: x['avg_prep_time'])

        print(f"\n{'='*60}")
        print(f"Fastest without preprocessing: {fastest_no_prep['engine']} ({fastest_no_prep['avg_no_prep_time']:.2f}s)")
        print(f"Fastest with preprocessing:    {fastest_prep['engine']} ({fastest_prep['avg_prep_time']:.2f}s)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
