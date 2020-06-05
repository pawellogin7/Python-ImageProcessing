import matplotlib.pyplot as plt
import numpy as np
import cv2

background_filename = 'dublin.jpg'
edited_filename = 'dublin_edited.jpg'

kernel = np.ones((5, 5), np.uint8)
background = cv2.imread(background_filename)
background_edited = cv2.imread(edited_filename)

background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
edited_gray = cv2.cvtColor(background_edited, cv2.COLOR_BGR2GRAY)
background_blur = cv2.GaussianBlur(background_gray, (5, 5), 0)
edited_blur = cv2.GaussianBlur(edited_gray, (5, 5), 0)

#Wycinanie templatea z obrazu
kevins_difference = np.square(edited_blur - background_blur)
ret_tresh, kevins_tresholded = cv2.threshold(kevins_difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kevins_denoised = cv2.morphologyEx(kevins_tresholded, cv2.MORPH_CLOSE, kernel)
kevins_denoised = cv2.morphologyEx(kevins_denoised, cv2.MORPH_OPEN, kernel)
kevins_where = np.where(kevins_denoised != 0, 1, 0)

kevins_separated = np.zeros(background_edited.shape)
kevins_separated[kevins_where == 1] = background_edited[kevins_where == 1]
kevins_separated = kevins_separated.astype(np.uint8)

kevins_contours, _ = cv2.findContours(kevins_tresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours_poly = [None]*len(kevins_contours)
bound_rect = [None]*len(kevins_contours)
for i, c in enumerate(kevins_contours):
    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    bound_rect[i] = cv2.boundingRect(contours_poly[i])

rect_max_area = 0
rect_max_area_id = 0
for i in range(len(kevins_contours)):
    current_area = cv2.contourArea(contours_poly[i], False)
    if current_area >= rect_max_area:
        rect_max_area = current_area
        rect_max_area_id = i


kevin_start_x, kevin_start_y, kevin_width, kevin_height = bound_rect[rect_max_area_id]
kevin_cropped = kevins_separated[kevin_start_y:kevin_start_y+kevin_height, kevin_start_x:kevin_start_x+kevin_width]


#Wycinanie wierszy i kolumn ze zbyt duza iloscia czarnych pixeli
crop_gray = cv2.cvtColor(kevin_cropped, cv2.COLOR_BGR2GRAY)
to_crop_up = 0
shape = crop_gray.shape
for i in range(crop_gray.shape[0]):
    nonzero_elements = np.count_nonzero(crop_gray[i, :])
    zero_elements = crop_gray.shape[1] - nonzero_elements
    if zero_elements >= nonzero_elements/2:
        to_crop_up += 1
    else:
        break

to_crop_down = 0
for i in range(crop_gray.shape[0]):
    nonzero_elements = np.count_nonzero(crop_gray[crop_gray.shape[0] - i - 1, :])
    zero_elements = crop_gray.shape[1] - nonzero_elements
    if zero_elements >= nonzero_elements/2:
        to_crop_down += 1
    else:
        break

if to_crop_down != 0:
    kevin_cropped_y = kevin_cropped[to_crop_up:-to_crop_down, :]
else:
    kevin_cropped_y = kevin_cropped[to_crop_up:, :]
crop_gray = cv2.cvtColor(kevin_cropped_y, cv2.COLOR_BGR2GRAY)

to_crop_left = 0
for i in range(crop_gray.shape[1]):
    nonzero_elements = np.count_nonzero(crop_gray[:, i])
    zero_elements = crop_gray.shape[0] - nonzero_elements
    if zero_elements >= nonzero_elements/2:
        to_crop_left += 1
    else:
        break

to_crop_right = 0
for i in range(crop_gray.shape[1]):
    nonzero_elements = np.count_nonzero(crop_gray[:, crop_gray.shape[1] - i - 1])
    zero_elements = crop_gray.shape[0] - nonzero_elements
    if zero_elements >= nonzero_elements/2:
        to_crop_down += 1
    else:
        break

if to_crop_right != 0:
    kevin_cropped_y_and_x = kevin_cropped_y[:, to_crop_left:-to_crop_right]
else:
    kevin_cropped_y_and_x = kevin_cropped_y[:, to_crop_left:]

#Zzajdowanie wartosci bound_low i bound_up maski HSV uzywanej przy szukaniu templata na obrazach
feature_matrix = kevin_cropped_y_and_x
feature_shape = feature_matrix.shape
hsv_feature = cv2.cvtColor(feature_matrix, cv2.COLOR_BGR2HSV)
hay = hsv_feature[:, :, 0]
saturation = hsv_feature[:, :, 1]
value = hsv_feature[:, :, 2]

hay_mean_value1 = np.mean(hay)
hay_mean_matrix1 = np.where(hay > hay_mean_value1, hay, 0)
hay_mean_matrix1 = hay_mean_matrix1[hay_mean_matrix1 != 0]
hay_mean_value2 = np.mean(hay_mean_matrix1)
hay_mean_matrix2 = np.where(hay > hay_mean_value2, hay, 0)
hay_mean_matrix2 = hay_mean_matrix1[hay_mean_matrix1 != 0]
hay_mean_value3 = np.mean(hay_mean_matrix2)
hay_low_val = int((hay_mean_value1 + hay_mean_value2 + hay_mean_value3) / 3)
hay_high_val = int(np.amax(hay))

sat_mean_value1 = np.mean(saturation)
sat_mean_matrix1 = np.where(saturation > sat_mean_value1, saturation, 0)
sat_mean_matrix1 = sat_mean_matrix1[sat_mean_matrix1 != 0]
sat_mean_value2 = np.mean(sat_mean_matrix1)
sat_mean_matrix2 = np.where(saturation > sat_mean_value2, saturation, 0)
sat_mean_matrix2 = sat_mean_matrix2[sat_mean_matrix2 != 0]
sat_mean_value3 = np.mean(sat_mean_matrix2)
sat_low_val = int((sat_mean_value1 + sat_mean_value2 + sat_mean_value3) / 3)
sat_high_val = int(np.amax(saturation))

val_mean_value1 = np.mean(value)
val_mean_matrix1 = np.where(value > val_mean_value1, value, 0)
val_mean_matrix1 = val_mean_matrix1[val_mean_matrix1 != 0]
val_mean_value2 = np.mean(val_mean_matrix1)
val_mean_matrix2 = np.where(value > val_mean_value2, value, 0)
val_mean_matrix2 = val_mean_matrix2[val_mean_matrix2 != 0]
val_mean_value3 = np.mean(val_mean_matrix2)
val_low_val = int((val_mean_value1 + val_mean_value2 + val_mean_value3) / 3)
val_high_val = int(np.amax(value))

bound_low = (hay_low_val, sat_low_val, val_low_val)
bound_up = (hay_high_val, sat_high_val, val_high_val)

#Szukanie templata na obrazach, znajdowanie najlepszego dopasowania
image_filenames = ["bodo.jpeg", "budapest.jpg", "ny.jpg", "prague.jpg", "rio.jpg"]
kevins_list = []
kevins_list.append(kevin_cropped)
for filename in image_filenames:
    image = cv2.imread(filename)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv_image, bound_low, bound_up)
    matching_result = cv2.bitwise_and(image, image, mask=hsv_mask)

    matching_result = cv2.cvtColor(matching_result, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(matching_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    bound_rect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        bound_rect[i] = cv2.boundingRect(contours_poly[i])

    max_area = 0
    max_area_id = 0
    for i in range(len(contours)):
        current_area = cv2.contourArea(contours_poly[i], False)
        if current_area >= max_area:
            max_area = current_area
            max_area_id = i

    rect_width = bound_rect[max_area_id][2]
    rect_height = bound_rect[max_area_id][3]
    points = contours[max_area_id][:, 0]
    point = contours[max_area_id][0]
    point = [np.mean(points[:, 0]), np.mean(point[:, 1])]

    if rect_height > rect_width:
        rect_x1 = int(point[0] - 5 * rect_width)
        rect_y1 = int(point[1] - 3 * rect_height)
        rect_x2 = int(point[0] + 4 * rect_width)
        rect_y2 = int(point[1] + 3 * rect_height)
    else:
        rect_x1 = int(point[0] - 5 * rect_height)
        rect_y1 = int(point[1] - 3 * rect_width)
        rect_x2 = int(point[0] + 4 * rect_height)
        rect_y2 = int(point[1] + 3 * rect_width)

    color = (0, 255, 0)
    kevins_list.append(image[rect_y1:rect_y2, rect_x1:rect_x2])
    image_bbox = image.copy()
    cv2.rectangle(image_bbox, (rect_x1, rect_y1), (rect_x2, rect_y2), color, 4)
    image_bbox = cv2.resize(image_bbox, (800, 600))
    cv2.imshow(filename, image_bbox)


if len(kevins_list) == 6:
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 8))
    ax1.imshow(cv2.cvtColor(kevins_list[0], cv2.COLOR_BGR2RGB))
    ax1.set_title("Kevin template")
    ax2.imshow(cv2.cvtColor(kevins_list[1], cv2.COLOR_BGR2RGB))
    ax2.set_title("Kevin in " + image_filenames[0])
    ax3.imshow(cv2.cvtColor(kevins_list[2], cv2.COLOR_BGR2RGB))
    ax3.set_title("Kevin in " + image_filenames[1])
    ax4.imshow(cv2.cvtColor(kevins_list[3], cv2.COLOR_BGR2RGB))
    ax4.set_title("Kevin in " + image_filenames[2])
    ax5.imshow(cv2.cvtColor(kevins_list[4], cv2.COLOR_BGR2RGB))
    ax5.set_title("Kevin in " + image_filenames[3])
    ax6.imshow(cv2.cvtColor(kevins_list[5], cv2.COLOR_BGR2RGB))
    ax6.set_title("Kevin in " + image_filenames[4])

plt.show()
cv2.waitKey()
