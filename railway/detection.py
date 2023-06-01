import cv2 as cv
import numpy as np

def detectRail(original):
    
    img = original.copy()

    img = cv.GaussianBlur(img, (0, 0), 0.5)

    img = cv.Sobel(img, -1, 1, 0, delta=50)

    img = cv.GaussianBlur(img, (0, 0), 4)

    img = cv.Canny(img, 10, 50)

    lines = cv.HoughLines(img, 1, np.pi/360, 200)

    return lines

if __name__ == '__main__':

    original = cv.resize(cv.imread("./Images/railway_7.jpg"), (1000, 600))

    lines = detectRail(original) # 이미지를 넣으면 감지된 선들을 반환함

    detected = original.copy()

    k = 5000 # 그냥 충분히 큰 값

    for line in lines: # 검출된 모든 선 순회

        r,theta = line[0] # 거리와 각도 # 가장 중요한 값!
        print(f"r: {r}, theta: {theta}")

        tx, ty = np.cos(theta), np.sin(theta)

        x0, y0 = r*tx, r*ty

        x1, y1 = int(x0 + k*(-ty)), int(y0 + k * tx) # 직선을 그려 줄 시작점과 끝점
        x2, y2 = int(x0 - k*(-ty)), int(y0 - k * tx)

        cv.line(detected, (x1, y1), (x2, y2), (0, 255, 0), 2) # 초록색 선 그려줌
    
    cv.imshow("Detected", detected)

    cv.waitKey(0) # 아무 키나 누르면 종료함