"""Configuration settings for the OCR receipt processing system."""

import os
from pathlib import Path


class Config:
    """Application configuration."""

    # Project paths
    BASE_DIR = Path(__file__).parent
    TEMP_DIR = BASE_DIR / "temp"
    OUTPUT_DIR = BASE_DIR / "output"

    # Create directories if they don't exist
    TEMP_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # OCR Settings
    DEFAULT_OCR_ENGINE = "paddleocr"  # tesseract, easyocr, paddleocr
    DEFAULT_PREPROCESSING = True
    OCR_CONFIDENCE_THRESHOLD = 0.5

    # LLM Settings
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    # Alternative models:
    # - Qwen/Qwen2.5-3B-Instruct (lighter, faster)
    # - Qwen/Qwen2.5-1.5B-Instruct (very light)
    # - Qwen/Qwen2.5-14B-Instruct (more accurate, heavier)

    USE_4BIT_QUANTIZATION = os.getenv("USE_4BIT_QUANTIZATION", "true").lower() == "true"
    LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "1024"))

    # API Settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_WORKERS = int(os.getenv("API_WORKERS", "1"))

    # File Processing
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    PDF_DPI = int(os.getenv("PDF_DPI", "300"))

    # Calculation
    CALCULATION_TOLERANCE = 0.02  # Allow 2 cents difference

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = "INFO"


# Select configuration based on environment
ENV = os.getenv("ENV", "development").lower()
if ENV == "production":
    config = ProductionConfig()
else:
    config = DevelopmentConfig()
