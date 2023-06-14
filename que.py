import threading
import socket
import time
from tkinter import *
from tkinter import messagebox
import cv2 as cv
import numpy as np
from queue import Queue

BUCKLING_TYPE1_TREHSHOLD = 10
BUCKLING_TYPE2_TREHSHOLD = 10 
BUCKLING_TYPE3_TREHSHOLD = 10
LINE_THRESHOLD = 150 
RESIZE_W = 1000
RESIZE_H = 600
isbuckling=True
isworking=True
latitude_location=37.528619
longitude_location=126.962837

def checkBuckling1(isbuckling):
    if isbuckling==True:
        text=f"Buckling Detected\nLocation: {latitude_location}, {longitude_location}"
        messagebox.showwarning("WARNING: type 1", text)
        isbuckling=False
def checkBuckling2(isbuckling):
    if isbuckling==True:
        text=f"Buckling Detected\nLocation: {latitude_location}, {longitude_location}"
        messagebox.showwarning("WARNING: type 1", text)
        isbuckling=False
def checkBuckling3(isbuckling):
    if isbuckling==True:
        text=f"Buckling Detected\nLocation: {latitude_location}, {longitude_location}"
        messagebox.showwarning("WARNING: type 1", text)
        isbuckling=False
def detectLines(original):
    
    lines = cv.HoughLines(original, 1, np.pi/360, LINE_THRESHOLD)

    return lines
def detectRail(original):

    img = original.copy()

    img = sobel(img)

    img = cv.GaussianBlur(img, (0, 0), 3.5)

    img = cv.Canny(img, 10, 50)

    return img
def sobel(original):

    img = original.copy()

    sobel1 = cv.Sobel(img, -1, 1, 0, delta=0)
    sobel2 = cv.Sobel(~img, -1, 1, 0, delta=0)
    img = cv.add(sobel1, sobel2)

    return img
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
    
    def detect(self, original, queue2):
        resized = cv.resize(original, (RESIZE_W, RESIZE_H))
        # transformed = transform(resized) # 시야(?)를 위에서 보는 것처럼 바꿈
        transformed = resized
        detected = detectRail(transformed) # 이미지를 넣으면 굵은 선들을 감지한 이미지를 반환함
        lines = detectLines(detected) # 이미지를 넣으면 감지된 선들을 반환함

        self.isBuckling = False
        thetaList = np.array([])
        drawn = transformed
        buk1 = False
        buk2 = False
        buk3 = False

        if lines is not None:
            for line in lines:
                r, theta = line[0]
                # print(f"r: {r}, theta: {theta}")
                drawn = drawLine(drawn, r, theta) # 감지된 선들을 이미지 위에 그려줌
                thetaList = np.append(thetaList, [theta])
        
            thetaList = parseList(thetaList)
            avgTheta = np.mean(thetaList)
            queue2.put(avgTheta)
            self.avgThetaList = np.append(self.avgThetaList[1:], np.array([avgTheta]))
            # self.isBuckling =  detectBucklingType1(avgTheta) or (detectBucklingType2(thetaList) or detectBucklingType3(self.avgThetaList))
            buk1 = detectBucklingType1(avgTheta)
            buk2 = detectBucklingType2(thetaList)
            buk3 = detectBucklingType3(self.avgThetaList)
           # return [drawn, self.isBuckling]
        return [drawn, buk1, buk2, buk3]
def total(queue,queue2):
    detector = Detector()
    while True:
        frame = queue.get()  # Get a frame from the queue
        if frame is not None:
            drawn, buk1, buk2, buk3 = detector.detect(frame, queue2)
            cv.imshow('shot', drawn)
            checkBuckling1(buk1)
            checkBuckling2(buk2)
            checkBuckling3(buk3)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
class MyTello:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for sending cmd
        self.socket.bind(("0.0.0.0", 8889))
        self.tello_address = ("192.168.10.1", 8889)
        self.receive_thread = threading.Thread(target=self.receive_thread)
        self.receive_thread.daemon = True
        self.receive_thread.start()
        self.socket.sendto('command'.encode('utf-8'), self.tello_address)
        print('sent: command')
        time.sleep(1)
        self.socket.sendto('takeoff'.encode('utf-8'), self.tello_address)
        
        time.sleep(8)
        self.socket.sendto('down 30'.encode('utf-8'), self.tello_address)
        time.sleep(4)
        self.socket.sendto('streamon'.encode('utf-8'), self.tello_address)
        time.sleep(1)
        print('sent: streamon\r\n')
        self._running = True
    
    def front(self):
        self.socket.sendto('forward 20'.encode('utf-8'), self.tello_address)

    def terminate(self):
        self._running = False
        self.video.release()
        cv.destroyAllWindows()

    def receive_thread(self):
        while True:
            try:
                self.response, ip = self.socket.recvfrom(3000)
                print(self.response)
            except socket.error as exc:
                print("Caught exception socket.error : %s" % exc)

    def recv(self,queue):
        self.video = cv.VideoCapture("udp://@0.0.0.0:11111")
        self.queue = queue
        while self._running:
            try:
                ret, frame = self.video.read()
                if ret:
                    cv.imshow('Tello', frame)
                if cv.waitKey (1)&0xFF == ord ('q'):
                    self.queue.put(frame)
            except Exception as err:
                print(err)

    def mystart(self, queue, queue2):
        self.queue = queue
        self.queue2 = queue2
        self.recv_videoThread = threading.Thread(target=self.recv, args=(self.queue,))
        self.recv_videoThread.daemon = True
        self.recv_videoThread.start()
        total_thread = threading.Thread(target=total, args=(self.queue,self.queue2))
        total_thread.daemon = True
        total_thread.start()
        cont_thread = threading.Thread(target=self.cont, args=(self.queue2,))
        cont_thread.daemon = True
        cont_thread.start()
        pass
    
    def cont(self, queue2):
        while True:
            the = queue2.get()  # Get a frame from the queue
            print(the)
            time.sleep(0.3)            
            if the is not None:
                if the >= 0:
                    a = f'ccw {the}'
                    self.socket.sendto(a.encode('utf-8'), self.tello_address)
                else:
                    a = f'cw {-the}'
                    self.socket.sendto(a.encode('utf-8'), self.tello_address)
                    print("ccw")
                time.sleep(1)
                self.socket.sendto('forward 20'.encode('utf-8'), self.tello_address)
                time.sleep(1)
    
    def myStop(self):
        print('myStop')
        self.socket.sendto('land'.encode('utf-8'), self.tello_address)
        pass
if __name__ == "__main__":
    print("Start program....")
    queue = Queue()
    queue2 = Queue()
    myTello = MyTello()
    myTello.mystart(queue,queue2)
    x=input()
    myTello.myStop()
    print("\n\nEnd of Program\n")