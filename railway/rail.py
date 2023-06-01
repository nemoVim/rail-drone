import cv2 as cv
import numpy as np
from line import Hough, HoughP
from utils import mergeGray2Color


def findRail(original):

    img = original.copy()

    img = cv.GaussianBlur(img, (0, 0), 0.5)

    img = cv.Sobel(img, -1, 1, 0, delta=50)

    img = cv.GaussianBlur(img, (0, 0), 4)

    img = cv.Canny(img, 10, 50)

    img = Hough(img, threshold = 200)
    # cv.imshow("Hough", img)

    # k = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    # img = cv.morphologyEx(img, cv.MORPH_CLOSE, k)

    return img

import time

if __name__ == "__main__":
    num = 7
    # original = cv.imread(f"./Images/railway_{num}.jpg", cv.IMREAD_GRAYSCALE)
    original = cv.imread(f"./Images/railway_{num}.jpg")

    original = cv.resize(original, (1000, 600))
    cv.imshow("Original img", original)

    t = time.time()
    img = findRail(original)
    print(time.time()-t)

    merged = mergeGray2Color(img, original)


    cv.imshow("Result img", merged)
    cv.imwrite(f"./Images/Result_{num}.jpg", img)

    a = cv.waitKey(0)