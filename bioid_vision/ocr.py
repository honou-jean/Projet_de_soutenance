"""Optical character recognition via EasyOCR, as described in the thesis
(Chapter 3.4, 3.8) and measured in Chapter 4.4 (88% character-level
accuracy on identity documents).
"""

import cv2
import easyocr
import numpy as np
from PIL import Image

from . import config


class TextExtractor:
    def __init__(self, languages=config.OCR_LANGUAGES, gpu=False):
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def extract_text(self, image_path):
        """Return the list of text fragments EasyOCR finds in an image file."""
        image = Image.open(image_path)
        img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        height, width = img.shape[:2]
        largest_side = max(height, width)
        if largest_side > config.OCR_MAX_IMAGE_DIMENSION:
            scale = config.OCR_MAX_IMAGE_DIMENSION / largest_side
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray)
        return [detection[1] for detection in results]
