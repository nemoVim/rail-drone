import cv2 as cv
import numpy as np

def Hough(original, degree = np.pi/360, threshold = 150):

    hough = np.zeros(original.shape, np.float32)

    a = 5000

    edge = original

    lines = cv.HoughLines(edge, 1, degree, threshold)

    # original.cvtColor(edge, edge, original.COLOR_GRAY2BGR)

    for line in lines: # 검출된 모든 선 순회
        r,theta = line[0] # 거리와 각도
        tx, ty = np.cos(theta), np.sin(theta) # x, y축에 대한 삼각비
        # x0, y0 = int(r/tx), int(r/ty)
        x0, y0 = r*tx, r*ty
        # cv.circle(original, (int(abs(x0)), int(abs(y0))), 3, (0,0,255))
        x1, y1 = int(x0 + a*(-ty)), int(y0 + a * tx)
        x2, y2 = int(x0 - a*(-ty)), int(y0 - a * tx)
        cv.line(hough, (x1, y1), (x2, y2), 1, 1)

    # merged = np.hstack((real_original, original))
    # print(hough.shape)
    # hough = cv.cvtColor(hough, cv.COLOR_BGR2GRAY)

    # cv.imshow('hough line', hough)

    return hough

def HoughP(original, degree = np.pi/360, threshold = 50):
    hough = np.zeros(original.shape, np.float32)
    lines = cv.HoughLinesP(original, 1, degree, threshold, minLineLength=20)
    for line in lines:
        cv.line(hough, (int(line[0][0]), int(line[0][1])), (int(line[0][2]), int(line[0][3])), 1, 1)
    
    return hough

if __name__ == "__main__":
    Hough("Canny", 3)
    k = cv.waitKey(0)