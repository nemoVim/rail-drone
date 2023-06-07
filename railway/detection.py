import cv2 as cv
import numpy as np
import time

def detectLines(original):
    
    lines = cv.HoughLines(original, 1, np.pi/180, 300)

    return lines

def detectRail_legacy_1(original):

    img = original.copy()

    img = cv.GaussianBlur(img, (0, 0), 0.5)

    img = cv.Sobel(img, -1, 1, 0, delta=50)

    img = cv.GaussianBlur(img, (0, 0), 4)

    img = cv.Canny(img, 10, 50)

    return img

def sobel(original):

    img = original.copy()

    sobel1 = cv.Sobel(img, -1, 1, 0, delta=0)
    sobel2 = cv.Sobel(~img, -1, 1, 0, delta=0)
    img = cv.add(sobel1, sobel2)

    return img

def thresholding(img, threshold):
    return (img > threshold) * img

def normalize(img):
    return np.uint8((img - np.min(img))*(255/(np.max(img)+np.min(img))))

def transform(original):

    p1 = [400, 0]
    p2 = [600, 0]
    p3 = [200, 600]
    p4 = [400, 600]

    p5 = [340, 0]
    p6 = [680, 0]
    trans = cv.getPerspectiveTransform(np.array([p1, p2, p3, p4], np.float32), np.array([p5, p6, p3, p4], np.float32))

    img = cv.warpPerspective(original.copy(), trans, (0, 0))
    
    return img


def a(img):
    img = cv.GaussianBlur(img, (0, 0), 3)
    img = normalize(img)
    img = thresholding(img, np.mean(img)*2)
    return img

def detectRail(original):
    
    # img = transform(original)
    img = original.copy()

    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv = (a(sobel(img)))
    h, s, v = cv.split(hsv)
    img = cv.GaussianBlur(img, (0, 0), 3)
    h, s, v = a(sobel(h)), a(sobel(s)), a(sobel(v))

    # cv.imshow("h", (h))
    # cv.imshow("s", (s))
    # cv.imshow("v", (v))

    img = cv.add(h, cv.add(s, v))

    img = cv.GaussianBlur(img, (0, 0), 3)
    k = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, k)
    img = thresholding(img, 100)
    img = cv.GaussianBlur(img, (0, 0), 3)
    img = cv.erode(img, k)
    img = thresholding(img, 150)
    img = cv.dilate(img, k)
    img = cv.dilate(img, k)
    img = thresholding(img, 200)
    img = cv.erode(img, k)

    return img 

def drawLines(original, lines):

    img = original.copy()

    k = 5000 # 그냥 충분히 큰 값

    if lines is None:
        return img

    for i, line in enumerate(lines): # 검출된 모든 선 순회

        r,theta = line[0] # 거리와 각도 # 가장 중요한 값!
        # print(f"r: {r}, theta: {theta}")

        tx, ty = np.cos(theta), np.sin(theta)

        x0, y0 = r*tx, r*ty

        x1, y1 = int(x0 + k*(-ty)), int(y0 + k * tx) # 직선을 그려 줄 시작점과 끝점
        x2, y2 = int(x0 - k*(-ty)), int(y0 - k * tx)

        # print(f"(x1, y1): ({x1}, {y1}), (x2, y2): ({x2}, {y2})")

        cv.line(img, (x1, y1), (x2, y2), (0, (i/len(lines))*255, 0), 2) # 초록색 선 그려줌

    return img

if __name__ == '__main__':

    # original = cv.resize(cv.imread("./Images/railway_10.jpg"), (1000, 600))
    original = cv.resize(cv.imread("./Images/railway.jpg"), (1000, 600))

    t_list = np.array([])

    for i in range(100):
        t = time.time()

        detected = detectRail(original) # 이미지를 넣으면 굵은 선들을 감지한 이미지를 반환함
        lines = detectLines(detected) # 이미지를 넣으면 감지된 선들을 반환함
        drawn = drawLines(original, lines) # 감지된 선들을 이미지 위에 그려줌

        t_list = np.append(t_list, [time.time()-t])


    print("time: ", np.mean(t_list))

    cv.imshow("Detected", detected)
    cv.imshow("Result", drawn)

    cv.waitKey(0) # 아무 키나 누르면 종료함