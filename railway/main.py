from detection import * 
from utils import mergeGray2Color
import cv2 as cv

cap = cv.VideoCapture("./Videos/hibiscus.mp4")

i = 0

fourcc = cv.VideoWriter_fourcc(*'DIVX')

out = cv.VideoWriter('rail.avi', fourcc, 30, (1000, 600))

detector = Detector()

while cap.isOpened():
    suc, frame = cap.read()

    if suc:

        print(i, end=": ")

        # if i == 100:
        #     frame = cv.resize(frame, (1000, 600))
        #     cv.imwrite("./Images/railway.jpg", frame)

        # if i > 500 and i < 800:
        if True:
            frame = cv.resize(frame, (1000, 600))
            # if i == 460:
            #     cv.imwrite("./Images/railway_light3.jpg", frame)
            # merged = mergeGray2Color(drawn, frame)
            # merged = cv.add(sobel(sobel(frame)), frame)
            # cv.imshow("Hibiscus", frame)
            drawn, buk1, buk2, buk3 = detector.detect(frame)
            cv.imshow("Hibiscus", drawn)
            print(buk1, buk2, buk3)
            if buk1 or (buk2 or buk3):
                cv.imwrite(f"./Images/buckling/buckling_{i}.jpg", drawn)

            # out.write(merged)
        
        # elif i >= 800:
        #     break

        i += 1

        key = cv.waitKey(1)

        if key == 27:
            break
    else:
        break

cap.release()
out.release()
cv.destroyAllWindows()