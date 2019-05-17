import label_util
import perspective
import feature_detection
import io_utils
import csv
import numpy as np
import cv2

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)

imname = "IMG_20190323_114634.jpg"
zaal = 13
path = "images\zalen\zaal_" + str(zaal) + "\\" + imname
print(path)
img = io_utils.imread(path)

with open("csv_corners\corners_" + str(zaal) + ".csv", mode = "r") as cornerlabels:
    label_reader = csv.reader(cornerlabels, delimiter=";")
    pts_lbl = np.array([])
    for row in label_reader:
        if len(row) >= 9 and row[0] == imname:
            pts_lbl = np.array([(int(row[1]), int(row[2])), (int(row[3]), int(row[4])),
                                (int(row[5]), int(row[6])), (int(row[7]), int(row[8]))])
    pts_det, _ = feature_detection.detect_perspective(img)
    if len(pts_det)>=4 and len(pts_lbl)>=4:
        label_util.draw_quad(pts_det, img, blue)
        label_util.draw_quad(pts_lbl, img, green)
        pts_int = label_util.get_intersection_pts(pts_lbl, pts_det)
        if len(pts_int)>=4:
            label_util.draw_quad(pts_int, img, red)
        height, width, depth = img.shape
        scale = height / 700
        img_scaled = cv2.resize(img, (int(width / scale), int(height / scale)))
        cv2.imshow(imname, img_scaled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif len(pts_det)<4 and len(pts_lbl)<4:
        print("Not enough points of both")
    elif len(pts_det)<4:
        print("Not enough points detected")
        print(pts_det)
    else:
        print("Not enough corners labeled")
        print(pts_lbl)
