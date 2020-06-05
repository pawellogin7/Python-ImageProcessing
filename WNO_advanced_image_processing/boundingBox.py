import cv2 as cv
import numpy as np

def thresh_callback(val):
    threshold = val

    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    contours, _ = cv.findContours(canny_output, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
        color = (0, 256, 0)
        cv.drawContours(drawing, contours_poly, i, color)
        cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])),
                     (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)

    cv.imshow('Contours', drawing)


src = cv.imread('./obrazki/dublin.jpg')
src_edited = cv.imread('./obrazki/dublin_edited.jpg')
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_edited_gray = cv.cvtColor(src_edited, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))
src_edited_gray = cv.blur(src_edited_gray, (3, 3))
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src_edited)

kernel = np.ones((3, 3), np.uint8)
src1 = src_edited - src
src1 = cv.dilate(src1, kernel, iterations=2)
src1 = cv.erode(src1, kernel, iterations=4)

max_thresh = 255
thresh = 100  # initial threshold
cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()