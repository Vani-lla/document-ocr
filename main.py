from os import environ

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path

TESS_PATH = "tess"
environ["TESSDATA_PREFIX"] = TESS_PATH


class DocumentOCR:
    def __init__(
        self,
        pdf_path: str,
        lang: str = "pol",
        config: str = "tessedit_char_blacklist=$|-vx»«",
    ):
        """
        Parameters:
            - lang : Language to use for Tesseract OCR. Default is "pol"
            - config : Tesseract configuration string. Default restricts certain characters

        Before calling the `get_text` method, ensure:
            - You have `TESSDATA_PREFIX` configured in your environment variables.
            - The necessary language models for Tesseract are downloaded.

        Usage:
            - Instantiate the class with the PDF path
            - Call the `get_text` method to get text in a single string
        """
        self.lang = lang
        self.config = config

        self.images = convert_from_path(pdf_path)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Increate contrast and brightness
        alpha, beta = 1.3, 0
        img = cv2.multiply(img, alpha)
        img = cv2.add(img, beta)

        # If image is noisy, denoise it
        if np.quantile(img, 0.30) < 240:
            img = cv2.fastNlMeansDenoising(img, None, 30.0, 7, 21)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # Binarization and further denoising
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 65
        )

        # Thicken font
        img = cv2.bitwise_not(img)
        img = cv2.dilate(img, (3, 3), iterations=1)

        # Get minimal rect to capture all black pixels
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(img))
        img = cv2.bitwise_not(img)

        return img[y : y + h, x : x + w]
        # return img

    def _get_boxes_from_image(self, image: np.ndarray) -> list[int, int, int, int]:
        # Blur and thicken everything
        img = cv2.GaussianBlur(image, (7, 7), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 15))
        dilate = cv2.dilate(cv2.bitwise_not(img), kernel, iterations=1)

        # Find bounding boxes
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        boxes = list(map(lambda x: cv2.boundingRect(x), cnts))

        # Filter out insignificant boxes
        boxes = filter(lambda box: box[3] > 30, boxes)

        return boxes

    def _get_list_from_tess_output(self, data: dict, min_conf: int) -> list[str]:
        texts = []
        for *_, conf, text in zip(*data.values()):
            if conf > min_conf and text.replace(" ", ""):
                texts.append(text)

        return texts

    def _deskew(self, image: np.ndarray, max_skew: int = 15) -> np.ndarray:
        h, w = image.shape

        coords = np.column_stack(np.where(cv2.bitwise_not(image) > 0))[:, ::-1]
        rect = cv2.minAreaRect(coords)
        if abs(rect[-1]) > max_skew:
            return image

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rect[-1], 1.0)

        return cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    def get_text(
        self, min_conf: int = 40, box_separator: str = "\n", debug_show: bool = False
    ) -> str:
        """
        Returns a string with text from all bounding boxes separated by a new line.

        Parameters:
            - min_conf : Minimum confidence for OCR text inclusion. Default is 40
            - box_separator : Separator for text from different bounding boxes. Default is newline.
        """
        data: list[str] = []
        for image in self.images:
            img = self._preprocess_image(np.array(image))
            img = self._deskew(img)
            
            if debug_show:
                cv2.imshow("x", img)
                cv2.waitKey()

            for x, y, w, h in self._get_boxes_from_image(img):
                box = img[y : y + h, x : x + w]

                box_data: dict = pytesseract.image_to_data(
                    box,
                    output_type=pytesseract.Output.DICT,
                    lang=self.lang,
                    config=self.config,
                )
                texts = self._get_list_from_tess_output(box_data, min_conf)

                if texts:
                    data.append(" ".join(texts))

        return box_separator.join(data)


if __name__ == "__main__":
    l = ["f2.pdf"]
    for path in l:
        ocr = DocumentOCR(f"pdfs/{path}")
        ocr.get_text(debug_show=True)
