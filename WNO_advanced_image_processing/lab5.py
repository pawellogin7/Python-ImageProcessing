import cv2
import numpy as np
from scipy import ndimage
from scipy import signal
import matplotlib.pyplot as plt

#------------------Opening main images--------------------
img = cv2.imread('./obrazki/dublin.jpg')
img_dublin = cv2.imread('./obrazki/dublin_edited.jpg')
img_edited = img_dublin.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray1 = cv2.cvtColor(img_edited, cv2.COLOR_BGR2GRAY)

#-----------Detecting objects im image and cutting the biggest one-----------------------
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
img1_blur = cv2.GaussianBlur(img_gray1, (5, 5), 0)

tresh_otzu, th = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
tresh_otzu1, th1 = cv2.threshold(img1_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
high_tresh_canny = tresh_otzu
low_tresh_canny = 0.5*tresh_otzu
high_tresh_canny1 = tresh_otzu1
low_tresh_canny1 = 0.5*tresh_otzu1

img_canny = cv2.Canny(img_gray, low_tresh_canny, high_tresh_canny)
img1_canny = cv2.Canny(img_gray1, low_tresh_canny1, high_tresh_canny1)
subs_canny = cv2.subtract(img1_canny, img_canny)

kernel = np.ones((5, 5), np.uint8)

subs_canny = cv2.dilate(subs_canny, kernel, iterations=8)
subs_canny = cv2.erode(subs_canny, kernel, iterations=14)
subs_canny = cv2.dilate(subs_canny, kernel, iterations=8)

contours, _ = cv2.findContours(subs_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours_poly = [None]*len(contours)
bound_rect = [None]*len(contours)
for i, c in enumerate(contours):
    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    bound_rect[i] = cv2.boundingRect(contours_poly[i])

max_area = 0
max_area_id = 0
img_bbox = img_edited.copy()
for i in range(len(contours)):
    current_area = cv2.contourArea(contours_poly[i], False)
    if current_area < 500*500:
        color = (0, 255, 0)
        cv2.rectangle(img_bbox, (int(bound_rect[i][0]), int(bound_rect[i][1])), (int(bound_rect[i][0] + bound_rect[i][2]),
                                    int(bound_rect[i][1] + bound_rect[i][3])), color, 2)
        if current_area >= max_area:
            max_area = current_area
            max_area_id = i



kevin_cropped = img_edited[int(bound_rect[max_area_id][1]):int(bound_rect[max_area_id][1] + bound_rect[max_area_id][3]),
           int(bound_rect[max_area_id][0]):int(bound_rect[max_area_id][0] + bound_rect[max_area_id][2])]
background_cropped = img[int(bound_rect[max_area_id][1]):int(bound_rect[max_area_id][1] + bound_rect[max_area_id][3]),
           int(bound_rect[max_area_id][0]):int(bound_rect[max_area_id][0] + bound_rect[max_area_id][2])]


#----------Removin background of image using watershed algorithm------------------------------
kevin_cropped = cv2.subtract(kevin_cropped, background_cropped)
kevin_gray = cv2.cvtColor(kevin_cropped, cv2.COLOR_BGR2GRAY)
kevin_gray = cv2.GaussianBlur(kevin_gray, (5, 5), 0)
tresh_otzu_kevin, th = cv2.threshold(kevin_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
high_tresh_canny = tresh_otzu_kevin
low_tresh_canny = 0.5*tresh_otzu_kevin
kevin_canny = cv2.Canny(kevin_gray, low_tresh_canny, high_tresh_canny)

kevin_bg = cv2.dilate(kevin_canny, kernel, iterations=6)
kevin_fg = cv2.dilate(kevin_canny, kernel, iterations=3)
kevin_fg = cv2.erode(kevin_fg, kernel, iterations=7)

_, markers = cv2.connectedComponents(kevin_fg)
markers = markers + 1
unknown = kevin_bg - kevin_fg
markers[unknown == 255] = 0

kevin_blur = cv2.GaussianBlur(kevin_cropped, (3, 3), 0)
markers = cv2.watershed(kevin_blur, markers)
kevin_cropped[markers == 1] = alpha = 1.0

kevin_cropped = kevin_cropped[0:70, 20:80]

kevin_resized = cv2.resize(kevin_cropped, (2*kevin_cropped.shape[1], 2*kevin_cropped.shape[0]), interpolation=cv2.INTER_LINEAR)


#----------------Searching for template on other images----------------------
img_rio = cv2.imread('./obrazki/rio.jpg')
img_rio = cv2.resize(img_rio, (int(img_rio.shape[1]/2), int(img_rio.shape[0]/2)), interpolation=cv2.INTER_LINEAR)

img_ny = cv2.imread('./obrazki/ny.jpg')
img_ny = cv2.resize(img_ny, (int(img_ny.shape[1]/3), int(img_ny.shape[0]/3)), interpolation=cv2.INTER_LINEAR)

img_back = img_rio.copy()
img_back_gray = cv2.cvtColor(img_back, cv2.COLOR_BGR2GRAY)

kevin_gray = cv2.cvtColor(kevin_cropped, cv2.COLOR_BGR2GRAY)
img_template_pyrr1 = kevin_gray.copy()
img_template_pyrr2 = cv2.pyrDown(img_template_pyrr1)
img_template_pyrr3 = cv2.pyrDown(img_template_pyrr2)
img_temp = img_template_pyrr3

img_back_blur = cv2.GaussianBlur(img_back_gray, (5, 5), 0)
img_temp_blur = cv2.GaussianBlur(img_temp, (5, 5), 0)

tresh_otzu, th = cv2.threshold(img_back_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
tresh_otzu1, th1 = cv2.threshold(img_temp_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
high_tresh_canny = tresh_otzu
low_tresh_canny = 0.5*tresh_otzu
high_tresh_canny1 = tresh_otzu1
low_tresh_canny1 = 0.5*tresh_otzu1

img_back_canny = cv2.Canny(img_back_blur, low_tresh_canny, high_tresh_canny)
img_temp_canny = cv2.Canny(img_temp_blur, low_tresh_canny1, high_tresh_canny1)

#image_filtered = signal.correlate2d(img_back_canny, img_temp_canny, boundary='symm', mode='same')
image_filtered = cv2.matchTemplate(img_back_canny, img_temp_canny, cv2.TM_CCORR_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image_filtered)
print(min_val, min_loc, max_val, max_loc)

# # Find template
# result = cv2.matchTemplate(img_back, img_temp, cv2.TM_CCOEFF_NORMED)
top_left = max_loc
h = img_temp.shape[0]
w = img_temp.shape[1]
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img_back, top_left, bottom_right,(0,0,255),2)


cv2.imshow('Harris', img_back)
cv2.imshow('Harris1', img_temp_canny)
cv2.imshow('Harris2', img_back_canny)
# cv2.imshow('Harris1', img1)
# cv2.imshow('Harris2', img2)


# cv2.imshow('Bounding boxy', img_bbox)
# cv2.imshow('Kevin', kevin_resized)
cv2.waitKey()
