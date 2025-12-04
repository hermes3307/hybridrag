import cv2
import numpy as np
from PIL import Image
from typing import Union
import logging

logger = logging.getLogger(__name__)


class ReceiptPreprocessor:
    """Image preprocessing for better OCR results on receipts."""

    @staticmethod
    def preprocess(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Apply preprocessing pipeline to receipt image.

        Args:
            image: Input image as numpy array or PIL Image

        Returns:
            Preprocessed image as numpy array
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply preprocessing steps
        processed = ReceiptPreprocessor._denoise(gray)
        processed = ReceiptPreprocessor._adjust_contrast(processed)
        processed = ReceiptPreprocessor._deskew(processed)
        processed = ReceiptPreprocessor._sharpen(processed)

        return processed

    @staticmethod
    def _denoise(image: np.ndarray) -> np.ndarray:
        """Remove noise from image."""
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    @staticmethod
    def _adjust_contrast(image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    @staticmethod
    def _deskew(image: np.ndarray) -> np.ndarray:
        """Correct skew in the image."""
        coords = np.column_stack(np.where(image > 0))
        if len(coords) == 0:
            return image

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Only deskew if angle is significant
        if abs(angle) < 0.5:
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    @staticmethod
    def _sharpen(image: np.ndarray) -> np.ndarray:
        """Sharpen the image for better OCR."""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def adaptive_threshold(image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for binarization."""
        return cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
