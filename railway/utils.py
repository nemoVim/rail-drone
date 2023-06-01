import cv2 as cv
import numpy as np

def mergeGray2Color(original_gray, original_color):
    gray = original_gray.copy()
    color = original_color.copy()

    img = cv.copyTo(np.uint8(np.full((600, 1000, 3), (0, 255, 0))), np.uint8(gray), color)

    # img = cv.add(color, gray)

    return img