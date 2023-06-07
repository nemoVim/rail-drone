from detection import * 
from utils import mergeGray2Color
import cv2 as cv

cap = cv.VideoCapture("./Videos/simulation.mp4")

i = 0

fourcc = cv.VideoWriter_fourcc(*'DIVX')

out = cv.VideoWriter('rail.avi', fourcc, 30, (1000, 600))

while cap.isOpened():
    suc, frame = cap.read()

    if suc:
        print(i)
        i += 1

        # if i == 100:
        #     frame = cv.resize(frame, (1000, 600))
        #     cv.imwrite("./Images/railway.jpg", frame)

        # if i > 500 and i < 800:
        if True:
            frame = cv.resize(frame, (1000, 600))
            # if i == 460:
            #     cv.imwrite("./Images/railway_light3.jpg", frame)
            detected = detectRail(frame) # 이미지를 넣으면 굵은 선들을 감지한 이미지를 반환함
            lines = detectLines(detected) # 이미지를 넣으면 감지된 선들을 반환함
            drawn = drawLines(frame, lines) # 감지된 선들을 이미지 위에 그려줌
            # merged = mergeGray2Color(drawn, frame)
            # merged = cv.add(sobel(sobel(frame)), frame)
            # cv.imshow("Hibiscus", drawn)
            cv.imshow("Hibiscus", drawn)

            # out.write(merged)
        
        # elif i >= 800:
        #     break

        key = cv.waitKey(1)

        if key == 27:
            break
    else:
        break

cap.release()
out.release()
cv.destroyAllWindows()