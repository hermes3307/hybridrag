import os
import tempfile
from typing import List, Union
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import logging

logger = logging.getLogger(__name__)


class FileHandler:
    """Utility class for handling different file formats."""

    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    SUPPORTED_PDF_FORMAT = {'.pdf'}

    @staticmethod
    def is_supported_format(file_path: str) -> bool:
        """Check if file format is supported."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in FileHandler.SUPPORTED_IMAGE_FORMATS or ext in FileHandler.SUPPORTED_PDF_FORMAT

    @staticmethod
    def load_image(file_path: str) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Load image from file path.

        Args:
            file_path: Path to image or PDF file

        Returns:
            Single image as numpy array or list of images for PDFs
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext in FileHandler.SUPPORTED_PDF_FORMAT:
            return FileHandler._load_pdf(file_path)
        elif ext in FileHandler.SUPPORTED_IMAGE_FORMATS:
            return FileHandler._load_image_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def _load_image_file(file_path: str) -> np.ndarray:
        """Load a single image file."""
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
        return image

    @staticmethod
    def _load_pdf(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
        """
        Convert PDF to images.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion

        Returns:
            List of images as numpy arrays
        """
        try:
            # Convert PDF to images
            pil_images = convert_from_path(pdf_path, dpi=dpi)

            # Convert PIL images to numpy arrays
            images = []
            for pil_img in pil_images:
                # Convert to RGB if needed
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')

                # Convert to numpy array (BGR for OpenCV)
                img_array = np.array(pil_img)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                images.append(img_array)

            logger.info(f"Converted PDF to {len(images)} image(s)")
            return images

        except Exception as e:
            logger.error(f"Failed to convert PDF: {e}")
            raise

    @staticmethod
    def save_image(image: np.ndarray, output_path: str):
        """Save image to file."""
        cv2.imwrite(output_path, image)
        logger.info(f"Image saved to {output_path}")

    @staticmethod
    def bytes_to_image(image_bytes: bytes) -> np.ndarray:
        """Convert bytes to numpy image array."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from bytes")
        return image

    @staticmethod
    def save_uploaded_file(file_bytes: bytes, filename: str) -> str:
        """
        Save uploaded file to temporary location.

        Args:
            file_bytes: File content as bytes
            filename: Original filename

        Returns:
            Path to saved file
        """
        ext = os.path.splitext(filename)[1].lower()

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        logger.info(f"Saved uploaded file to {tmp_path}")
        return tmp_path
