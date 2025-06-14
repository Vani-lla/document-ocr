from glob import glob
from json import dump
from os import environ, remove

import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
from pdf2image import convert_from_path
from PIL import Image

from helpers import boxes_from_image

environ["TESSDATA_PREFIX"] = "tess"


def preprocess(image, save=False) -> Image:
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    alpha, beta = 1.3, 0

    img = cv2.multiply(img, alpha)
    img = cv2.add(img, beta)

    if np.quantile(img, 0.30) < 240:
        img = cv2.fastNlMeansDenoising(img, None, 30.0, 7, 21)

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 65
    )

    img = cv2.bitwise_not(img)
    img = cv2.dilate(img, (3, 3), iterations=1)
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(img))
    img = cv2.bitwise_not(img)

    return img[y : y + h, x : x + w]


fig = plt.figure(6, layout="tight")
for i in range(1, 6):
    images = convert_from_path(f"pdfs/f{i}.pdf")
    for p in glob(f"pdfs/f{i}/box*.png"):
        remove(p)

    img = np.array(images[0])
    img = preprocess(img)

    data = {}
    for ind, (x, y, w, h) in enumerate(boxes_from_image(img)):
        box = img[y : y + h, x : x + w]
        cv2.imwrite(f"pdfs/f{i}/box{ind}.png", box)

        box_data: dict = pytesseract.image_to_data(
            box,
            output_type=pytesseract.Output.DICT,
            lang="pol",
            config=r"tessedit_char_blacklist=$|-vx»«",
        )
        texts: list[str] = []
        for (
            level,
            page_num,
            block_num,
            par_num,
            line_num,
            word_num,
            left,
            top,
            width,
            height,
            conf,
            text,
        ) in zip(*box_data.values()):
            if conf > 40 and text.replace(" ", ""):
                texts.append(text)
        if texts:
            data[ind] = " ".join(texts)

    with open(f"pdfs/f{i}/data.json", "w", encoding="utf-8") as file:
        dump(data, file, ensure_ascii=False, indent=4)

    with open(f"pdfs/f{i}/data.txt", "w", encoding="utf-8") as file:
        file.writelines(map(lambda l: l + "\n", data.values()))

    cv2.imwrite(f"pdfs/images/i{i}.jpg", img)

    ax = fig.add_subplot(2, 3, i)
    ax.imshow(img, cmap="grey")
    ax.set_axis_off()


plt.show()
