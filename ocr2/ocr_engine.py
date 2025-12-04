import cv2
import numpy as np
from PIL import Image
from typing import Union, Optional
import logging
from models import OCREngine
from preprocessor import ReceiptPreprocessor

logger = logging.getLogger(__name__)


class OCRService:
    """Service for performing OCR on receipt images with multiple engine support."""

    def __init__(self):
        self._tesseract_available = False
        self._easyocr_reader = None
        self._paddleocr_reader = None
        self.preprocessor = ReceiptPreprocessor()

    def _init_tesseract(self):
        """Initialize Tesseract OCR."""
        if not self._tesseract_available:
            try:
                import pytesseract
                self._tesseract_available = True
                logger.info("Tesseract OCR initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Tesseract: {e}")
                raise

    def _init_easyocr(self):
        """Initialize EasyOCR."""
        if self._easyocr_reader is None:
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(['en'], gpu=True)
                logger.info("EasyOCR initialized")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                # Fallback to CPU
                try:
                    self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
                    logger.info("EasyOCR initialized (CPU mode)")
                except Exception as e2:
                    logger.error(f"Failed to initialize EasyOCR on CPU: {e2}")
                    raise

    def _init_paddleocr(self):
        """Initialize PaddleOCR."""
        if self._paddleocr_reader is None:
            try:
                from paddleocr import PaddleOCR
                self._paddleocr_reader = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=True,
                    show_log=False
                )
                logger.info("PaddleOCR initialized")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR with GPU: {e}")
                # Fallback to CPU
                try:
                    from paddleocr import PaddleOCR
                    self._paddleocr_reader = PaddleOCR(
                        use_angle_cls=True,
                        lang='en',
                        use_gpu=False,
                        show_log=False
                    )
                    logger.info("PaddleOCR initialized (CPU mode)")
                except Exception as e2:
                    logger.error(f"Failed to initialize PaddleOCR on CPU: {e2}")
                    raise

    def extract_text(
        self,
        image: Union[np.ndarray, Image.Image, str],
        engine: OCREngine = OCREngine.PADDLEOCR,
        preprocess: bool = True
    ) -> str:
        """
        Extract text from image using specified OCR engine.

        Args:
            image: Input image (numpy array, PIL Image, or file path)
            engine: OCR engine to use
            preprocess: Whether to apply preprocessing

        Returns:
            Extracted text
        """
        # Load image if path provided
        if isinstance(image, str):
            image = cv2.imread(image)

        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Preprocess if requested
        if preprocess:
            image = self.preprocessor.preprocess(image)

        # Perform OCR based on engine
        if engine == OCREngine.TESSERACT:
            return self._ocr_tesseract(image)
        elif engine == OCREngine.EASYOCR:
            return self._ocr_easyocr(image)
        elif engine == OCREngine.PADDLEOCR:
            return self._ocr_paddleocr(image)
        else:
            raise ValueError(f"Unsupported OCR engine: {engine}")

    def _ocr_tesseract(self, image: np.ndarray) -> str:
        """Perform OCR using Tesseract."""
        self._init_tesseract()
        import pytesseract

        # Configure Tesseract for receipt OCR
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
        return text.strip()

    def _ocr_easyocr(self, image: np.ndarray) -> str:
        """Perform OCR using EasyOCR."""
        self._init_easyocr()

        results = self._easyocr_reader.readtext(image)
        text = '\n'.join([result[1] for result in results])
        return text.strip()

    def _ocr_paddleocr(self, image: np.ndarray) -> str:
        """Perform OCR using PaddleOCR."""
        self._init_paddleocr()

        results = self._paddleocr_reader.ocr(image, cls=True)

        if results is None or len(results) == 0:
            return ""

        text_lines = []
        for line in results[0]:
            if line:
                text_lines.append(line[1][0])

        return '\n'.join(text_lines).strip()
