# trans = cv.getPerspectiveTransform(np.array([[1, 2], [3, 4], [5, 6], [7, 8]], np.float32), np.array([[9, 10], [11, 12], [13, 14], [15,16]], np.float32))

# transformed = cv.warpPerspective(detected, trans, (0, 0))