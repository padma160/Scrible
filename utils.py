import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import numpy as np
import easyocr
import os

# Initialize OCR reader
reader = None

def get_ocr_reader():
    """Initialize and return the OCR reader"""
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)
    return reader


def enhance_image(image):
    """Improve image quality for better OCR"""
    
    # Convert to grayscale
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Increase brightness and contrast
    bright = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)

    # Apply thresholding
    thresh = cv2.threshold(
        bright, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return thresh


def extract_text(image):
    """Extract text from image using EasyOCR"""
    
    reader = get_ocr_reader()
    results = reader.readtext(image)

    # Combine detected text
    text = "\n".join([result[1] for result in results])

    return text
