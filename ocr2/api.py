from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import os
import time
import logging
from typing import Optional, List
import uvicorn

from models import (
    OCREngine, OCRResponse, ReceiptSummary,
    BatchReceiptResponse, BatchReceiptResult, AggregatedTotals
)
from ocr_engine import OCRService
from llm_parser import Qwen3ReceiptParser
from calculator import ReceiptCalculator
from aggregator import ReceiptAggregator
from utils import FileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OCR Receipt Processing API",
    description="API for OCR-based receipt processing with LLM parsing and calculation",
    version="1.0.0"
)

# Global services
ocr_service = OCRService()
llm_parser = None  # Lazy load due to model size
calculator = ReceiptCalculator()
aggregator = ReceiptAggregator()


def get_llm_parser():
    """Lazy load LLM parser."""
    global llm_parser
    if llm_parser is None:
        logger.info("Initializing LLM parser (this may take a moment)...")
        llm_parser = Qwen3ReceiptParser(
            model_name="Qwen/Qwen2.5-7B-Instruct",  # Can be changed to 3B or smaller
            use_4bit=True
        )
    return llm_parser


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "OCR Receipt Processing API",
        "version": "1.0.0",
        "endpoints": {
            "process_receipt": "/api/v1/process-receipt",
            "process_receipt_simple": "/api/v1/process-receipt-simple",
            "process_receipts_batch": "/api/v1/process-receipts-batch",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/api/v1/process-receipt", response_model=OCRResponse)
async def process_receipt(
    file: UploadFile = File(...),
    ocr_engine: str = Form("paddleocr"),
    preprocess: bool = Form(True),
    use_llm: bool = Form(True)
):
    """
    Process a receipt image or PDF.

    Args:
        file: Receipt file (PDF, JPG, PNG, etc.)
        ocr_engine: OCR engine to use (tesseract, easyocr, paddleocr)
        preprocess: Whether to apply image preprocessing
        use_llm: Whether to use LLM for structured parsing

    Returns:
        OCRResponse with extracted text and structured receipt data
    """
    start_time = time.time()
    temp_file_path = None

    try:
        # Validate OCR engine
        try:
            engine = OCREngine(ocr_engine.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid OCR engine. Must be one of: {[e.value for e in OCREngine]}"
            )

        # Read uploaded file
        file_bytes = await file.read()
        temp_file_path = FileHandler.save_uploaded_file(file_bytes, file.filename)

        # Check if supported format
        if not FileHandler.is_supported_format(temp_file_path):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported: {FileHandler.SUPPORTED_IMAGE_FORMATS | FileHandler.SUPPORTED_PDF_FORMAT}"
            )

        # Load image(s)
        images = FileHandler.load_image(temp_file_path)

        # Handle PDF with multiple pages
        if isinstance(images, list):
            logger.info(f"Processing {len(images)} page(s) from PDF")
            # Process first page for now (can be extended to handle multiple pages)
            image = images[0]
            if len(images) > 1:
                logger.warning(f"PDF has {len(images)} pages. Processing only the first page.")
        else:
            image = images

        # Perform OCR
        logger.info(f"Performing OCR with engine: {engine.value}")
        raw_text = ocr_service.extract_text(image, engine=engine, preprocess=preprocess)

        if not raw_text:
            return OCRResponse(
                success=False,
                raw_text="",
                receipt_summary=None,
                processing_time=time.time() - start_time,
                ocr_engine_used=engine.value,
                error="No text extracted from image"
            )

        logger.info(f"Extracted {len(raw_text)} characters")

        # Parse with LLM if requested
        receipt_summary = None
        if use_llm:
            logger.info("Parsing receipt with LLM...")
            try:
                parser = get_llm_parser()
                receipt_summary = parser.parse_receipt(raw_text)

                if receipt_summary:
                    # Validate and calculate totals
                    receipt_summary = calculator.validate_and_calculate(receipt_summary)
                    logger.info("Receipt parsed and validated successfully")
                else:
                    logger.warning("LLM parsing returned no results")

            except Exception as e:
                logger.error(f"LLM parsing failed: {e}")
                # Continue without LLM parsing

        processing_time = time.time() - start_time

        return OCRResponse(
            success=True,
            raw_text=raw_text,
            receipt_summary=receipt_summary,
            processing_time=processing_time,
            ocr_engine_used=engine.value
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing receipt: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temp file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file: {e}")


@app.post("/api/v1/process-receipt-simple")
async def process_receipt_simple(
    file: UploadFile = File(...),
    ocr_engine: str = Form("paddleocr")
):
    """
    Simple receipt processing without LLM (faster, less resource-intensive).

    Args:
        file: Receipt file
        ocr_engine: OCR engine to use

    Returns:
        Raw OCR text
    """
    temp_file_path = None

    try:
        # Validate OCR engine
        try:
            engine = OCREngine(ocr_engine.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid OCR engine. Must be one of: {[e.value for e in OCREngine]}"
            )

        # Read uploaded file
        file_bytes = await file.read()
        temp_file_path = FileHandler.save_uploaded_file(file_bytes, file.filename)

        # Load image(s)
        images = FileHandler.load_image(temp_file_path)

        if isinstance(images, list):
            image = images[0]
        else:
            image = images

        # Perform OCR
        raw_text = ocr_service.extract_text(image, engine=engine, preprocess=True)

        return {"text": raw_text, "engine": engine.value}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing receipt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass


@app.post("/api/v1/process-receipts-batch", response_model=BatchReceiptResponse)
async def process_receipts_batch(
    files: List[UploadFile] = File(...),
    ocr_engine: str = Form("paddleocr"),
    preprocess: bool = Form(True),
    use_llm: bool = Form(True)
):
    """
    Process multiple receipt images or PDFs in batch and return aggregated totals.

    Args:
        files: List of receipt files (PDF, JPG, PNG, etc.)
        ocr_engine: OCR engine to use (tesseract, easyocr, paddleocr)
        preprocess: Whether to apply image preprocessing
        use_llm: Whether to use LLM for structured parsing

    Returns:
        BatchReceiptResponse with individual results and aggregated totals
    """
    start_time = time.time()
    temp_file_paths = []
    results = []

    try:
        # Validate OCR engine
        try:
            engine = OCREngine(ocr_engine.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid OCR engine. Must be one of: {[e.value for e in OCREngine]}"
            )

        logger.info(f"Processing {len(files)} receipts in batch mode")

        # Process each file
        for file in files:
            temp_file_path = None
            result = BatchReceiptResult(filename=file.filename, success=False)

            try:
                # Read uploaded file
                file_bytes = await file.read()
                temp_file_path = FileHandler.save_uploaded_file(file_bytes, file.filename)
                temp_file_paths.append(temp_file_path)

                # Check if supported format
                if not FileHandler.is_supported_format(temp_file_path):
                    result.error = f"Unsupported file format"
                    results.append(result)
                    continue

                # Load image(s)
                images = FileHandler.load_image(temp_file_path)

                # Handle PDF with multiple pages
                if isinstance(images, list):
                    image = images[0]
                    if len(images) > 1:
                        logger.warning(f"{file.filename}: Processing only first page of {len(images)} pages")
                else:
                    image = images

                # Perform OCR
                raw_text = ocr_service.extract_text(image, engine=engine, preprocess=preprocess)

                if not raw_text:
                    result.error = "No text extracted from image"
                    results.append(result)
                    continue

                result.raw_text = raw_text

                # Parse with LLM if requested
                if use_llm:
                    try:
                        parser = get_llm_parser()
                        receipt_summary = parser.parse_receipt(raw_text)

                        if receipt_summary:
                            # Validate and calculate totals
                            receipt_summary = calculator.validate_and_calculate(receipt_summary)
                            result.receipt_summary = receipt_summary
                            result.success = True
                        else:
                            result.error = "LLM parsing returned no results"

                    except Exception as e:
                        logger.error(f"LLM parsing failed for {file.filename}: {e}")
                        result.error = f"LLM parsing failed: {str(e)}"
                else:
                    # Without LLM, just mark as success with raw text
                    result.success = True

            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                result.error = str(e)

            results.append(result)

        # Aggregate results
        aggregated_totals = aggregator.aggregate_receipts(results)

        processing_time = time.time() - start_time

        logger.info(
            f"Batch processing completed: {aggregated_totals.successful_receipts}/{len(files)} successful, "
            f"Grand total: ${aggregated_totals.grand_total:.2f}"
        )

        return BatchReceiptResponse(
            total_processing_time=processing_time,
            ocr_engine_used=engine.value,
            individual_results=results,
            aggregated_totals=aggregated_totals
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp files
        for temp_file_path in temp_file_paths:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_file_path}: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
