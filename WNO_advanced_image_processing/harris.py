import cv2
import numpy as np
from scipy import ndimage
from scipy import signal
import matplotlib.pyplot as plt

#--------------Funkcje----------------------
def distance_calculate(point1, point2):
    distance = np.sqrt(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1]))
    return distance

img = cv2.imread("./obrazki/kevin.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template_shape = gray.shape

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Threshold for an optimal value, it may vary depending on the image.
ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

# Now draw them
res = np.hstack((centroids, corners))
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]

# size = int(np.amax([template_shape[0], template_shape[1]]) / 5) // 2 * 2 + 1
size = 5
features = np.empty((res.shape[0], size, size))
features_points = np.empty((res.shape[0], 2))
for i in range(res.shape[0]):
    x = res[i, 0]
    y = res[i, 1]
    features_points[i, 0] = x
    features_points[i, 1] = y
    gray1 = np.zeros((gray.shape[0]+size*2, gray.shape[1]+size*2))
    gray1[size:-size, size:-size] = gray
    features[i] = gray1[y-size//2+size:y+size//2+1+size, x-size//2+size:x+size//2+1+size]

back = cv2.imread("./obrazki/dublin_edited.jpg")
back_gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
matches = np.empty((features.shape[0], 2))
for i in range(features.shape[0]):
    template = features[i].astype(np.uint8)
    a = (back_gray - np.mean(back_gray)) / (np.std(back_gray) * len(back_gray))
    b = (template - np.mean(template)) / (np.std(template))
    corr = signal.correlate2d(a, b, boundary='symm', mode='same')
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)
    matches[i, 0] = max_loc[0]
    matches[i, 1] = max_loc[1]
    back[max_loc[1], max_loc[0]] = (0, 0, 255)



distance = np.empty((matches.shape[0], matches.shape[0]))
for i in range(matches.shape[0]):
    for j in range(matches.shape[0]):
        distance[i, j] = np.sqrt(np.power(matches[j, 0]-matches[i, 0], 2) + np.power(matches[j, 1]-matches[i, 1], 2))

distance = np.where(distance <= np.max([template_shape[0], template_shape[1]]), distance, 0)
max_nonzero = 0
max_nonzero_id = 0
for i in range(distance.shape[0]):
    dist_vect = distance[i]
    nonzero_values = np.sum(np.nonzero(dist_vect))
    if nonzero_values > max_nonzero:
        max_nonzero_id = i
        max_nonzero = nonzero_values

good_distance_vector = distance[max_nonzero_id]
good_matches_id = []
good_matches_id.append(max_nonzero_id)
for i in range(good_distance_vector.shape[0]):
    if(good_distance_vector[i] != 0):
        good_matches_id.append(i)

good_matches = np.empty((len(good_matches_id), 2))
matches_indexes = np.arange(len(good_matches_id))
good_matches[matches_indexes, 0] = matches[good_matches_id, 0]
good_matches[matches_indexes, 1] = matches[good_matches_id, 1]


temp_x_min = int(np.amin(features_points[:, 0]))
temp_x_max = int(np.amax(features_points[:, 0]))
temp_y_min = int(np.amin(features_points[:, 1]))
temp_y_max = int(np.amax(features_points[:, 1]))

back_x_min = int(np.amin(good_matches[:, 0]))
back_x_max = int(np.amax(good_matches[:, 0]))
back_y_min = int(np.amin(good_matches[:, 1]))
back_y_max = int(np.amax(good_matches[:, 1]))

delta_x_back = back_x_max - back_x_min
delta_y_back = back_y_max - back_y_min
delta_x_temp = temp_x_max - temp_x_min
delta_y_temp = temp_y_max - temp_y_min
if delta_y_back/delta_x_back >= 1 and delta_y_temp/delta_x_temp >= 1 or \
        delta_y_back/delta_x_back <= 1 and delta_y_temp/delta_x_temp <= 1:
    scale = np.amax([delta_x_back, delta_y_back]) / np.amax([delta_x_temp, delta_y_temp])
    temp_height = template_shape[0]
    temp_width = template_shape[1]
    bbox_x_shift = (temp_width * scale - delta_x_back) / 2
    bbox_y_shift = (temp_height * scale - delta_y_back) / 2
    bbox_width = temp_width * scale
    bbox_height = temp_height * scale

else:
    scale = np.amax([delta_x_back, delta_y_back]) / np.amax([delta_x_temp, delta_y_temp])
    temp_height = template_shape[1]
    temp_width = template_shape[0]
    bbox_x_shift = (temp_width * scale - delta_x_back) / 2
    bbox_y_shift = (temp_height * scale - delta_y_back) / 2
    bbox_width = temp_width * scale
    bbox_height = temp_height * scale


print(scale)
# bbox_top_left = (int(np.amin(good_matches[:, 0]) - size//2), int(np.amin(good_matches[:, 1]) - size//2))
# bbox_bottom_right = (int(np.amax(good_matches[:, 0]) + size), int(np.amax(good_matches[:, 1]) + size))
bbox_top_left = (int(np.amin(good_matches[:, 0]) - bbox_x_shift), int(np.amin(good_matches[:, 1]) - bbox_y_shift))
bbox_bottom_right = (int(bbox_top_left[0] + bbox_width), int(bbox_top_left[1] + bbox_height))


cv2.rectangle(back, bbox_top_left, bbox_bottom_right, (0, 255, 0), 1)


cv2.imshow('dst1', cv2.resize(back, (back.shape[1], back.shape[0])))
cv2.imshow('dst', cv2.resize(img, (img.shape[1]*3, img.shape[0]*3)))
cv2.waitKey()

