import cv2 as cv
import numpy as np

BUCKLING_TYPE1_TREHSHOLD = 10
BUCKLING_TYPE2_TREHSHOLD = 10 
BUCKLING_TYPE3_TREHSHOLD = 10
LINE_THRESHOLD = 150 
RESIZE_W = 1000
RESIZE_H = 600


def detectLines(original):
    
    lines = cv.HoughLines(original, 1, np.pi/360, LINE_THRESHOLD)

    return lines

def detectRail(original):

    img = original.copy()

    img = sobel(img)

    img = cv.GaussianBlur(img, (0, 0), 3.5)

    img = cv.Canny(img, 10, 50)

    return img

def detectRail_legacy_1(original):

    # img = transform(original)
    # img = img.copy()
    img = original.copy()

    img = cut(sobel(img))
    # cv.imshow("sobel", img)

    img = cv.GaussianBlur(img, (0, 0), 4)

    img = cv.Canny(img, 10, 50)
    # cv.imshow("canny", img)

    k = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, k)
    img = cv.GaussianBlur(img, (0, 0), 4)
    img = thresholding(img, 100)
    k = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    img = cv.erode(img, k)
    img = thresholding(img, 200)
    cv.imshow("close", img)

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

    p5 = [345, 0]
    p6 = [665, 0]
    trans = cv.getPerspectiveTransform(np.array([p1, p2, p3, p4], np.float32), np.array([p5, p6, p3, p4], np.float32))

    img = cv.warpPerspective(original.copy(), trans, (0, 0))
    
    return img

def cut(img):
    img = cv.GaussianBlur(img, (0, 0), 3)
    img = normalize(img)
    img = thresholding(img, np.mean(img)*2)
    return img

def detectRail_legacy(original):
    
    img = original.copy()

    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv = (cut(sobel(img)))
    h, s, v = cv.split(hsv)
    img = cv.GaussianBlur(img, (0, 0), 3)
    h, s, v = cut(sobel(h)), cut(sobel(s)), cut(sobel(v))

    # cv.imshow("h", (h))
    # cv.imshow("s", (s))
    # cv.imshow("v", (v))

    img = cv.add(h, cv.add(s, v))

    # img = cv.GaussianBlur(img, (0, 0), 3)
    # k = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    # img = cv.morphologyEx(img, cv.MORPH_CLOSE, k)
    # img = thresholding(img, 100)
    # img = cv.GaussianBlur(img, (0, 0), 3)
    # img = cv.erode(img, k)

    # img = thresholding(img, 150)
    # img = cv.dilate(img, k)
    # img = cv.dilate(img, k)
    # img = thresholding(img, 200)
    # img = cv.erode(img, k)

    return img 

def drawLine(original, r, theta):

    img = original.copy()

    k = 5000 # 그냥 충분히 큰 값

    tx, ty = np.cos(theta), np.sin(theta)

    x0, y0 = r*tx, r*ty

    x1, y1 = int(x0 + k*(-ty)), int(y0 + k * tx) # 직선을 그려 줄 시작점과 끝점
    x2, y2 = int(x0 - k*(-ty)), int(y0 - k * tx)
    # print(f"(x1, y1): ({x1}, {y1}), (x2, y2): ({x2}, {y2})")

    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # 초록색 선 그려줌

    return img

def detectBucklingType1(avgTheta):
    return avgTheta >= BUCKLING_TYPE1_TREHSHOLD

def parseList(list):
    list = np.round(list * 180 / np.pi)
    list = list - 180*(list//90)
    return list

def getDelta(valueList):
    maxValue = np.max(valueList)
    minValue = np.min(valueList)
    return maxValue - minValue

def detectBucklingType2(thetaList):
    return getDelta(thetaList) >= BUCKLING_TYPE2_TREHSHOLD

def detectBucklingType3(avgThetaList):
    return getDelta(avgThetaList) >= BUCKLING_TYPE3_TREHSHOLD

class Detector:
    def __init__(self):
        self.avgThetaList = np.array([0, 0, 0, 0, 0])
        self.isBuckling = False
    
    def detect(self, original):
        resized = cv.resize(original, (RESIZE_W, RESIZE_H))
        transformed = transform(resized) # 시야(?)를 위에서 보는 것처럼 바꿈
        detected = detectRail(transformed) # 이미지를 넣으면 굵은 선들을 감지한 이미지를 반환함
        lines = detectLines(detected) # 이미지를 넣으면 감지된 선들을 반환함

        self.isBuckling = False
        thetaList = np.array([])
        drawn = transformed

        if lines is not None:
            for line in lines:
                r, theta = line[0]
                # print(f"r: {r}, theta: {theta}")
                drawn = drawLine(drawn, r, theta) # 감지된 선들을 이미지 위에 그려줌
                thetaList = np.append(thetaList, [theta])
        
            thetaList = parseList(thetaList)
            avgTheta = np.mean(thetaList)
            self.avgThetaList = np.append(self.avgThetaList[1:], np.array([avgTheta]))
            # self.isBuckling =  detectBucklingType1(avgTheta) or (detectBucklingType2(thetaList) or detectBucklingType3(self.avgThetaList))
            buk1 = detectBucklingType1(avgTheta)
            buk2 = detectBucklingType2(thetaList)
            buk3 = detectBucklingType3(self.avgThetaList)
        

        # return [drawn, self.isBuckling]
        return [drawn, buk1, buk2, buk3]

if __name__ == '__main__':

    original = cv.imread("./Images/railway.jpg")
    detector = Detector()
    # drawn, isBuckling = detector.detect(original)
    # print(isBuckling)
    drawn, buk1, buk2, buk3 = detector.detect(original)
    cv.imshow("Result", drawn)
    print(buk1, buk2, buk3)

    cv.waitKey(0) # 아무 키나 누르면 종료함