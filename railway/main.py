from rail import findRail
from utils import mergeGray2Color
import cv2 as cv

cap = cv.VideoCapture("./Videos/hibiscus.mp4")

i = 0

fourcc = cv.VideoWriter_fourcc(*'DIVX')

out = cv.VideoWriter('rail.avi', fourcc, 30, (1000, 600))

while cap.isOpened():
    suc, frame = cap.read()

    if suc:
        print(i)
        i += 1
        if i > 500 and i < 800:
            frame = cv.resize(frame, (1000, 600))
            img = findRail(frame)
            merged = mergeGray2Color(img, frame)
            cv.imshow("Hibiscus", merged)

            # out.write(merged)

        key = cv.waitKey(1)

        if key == 27:
            break
    else:
        break

cap.release()
out.release()
cv.destroyAllWindows()