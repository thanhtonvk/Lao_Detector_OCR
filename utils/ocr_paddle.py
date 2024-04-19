from paddleocr import PaddleOCR
import numpy as np
import cv2
import re


class PaddleReader:
    def __init__(self) -> None:
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def read(self, image: np.ndarray) -> str:
        result = self.paddle_ocr.ocr(image, det=False, rec=True)

        return result


if __name__ == "__main__":
    reader = PaddleReader()