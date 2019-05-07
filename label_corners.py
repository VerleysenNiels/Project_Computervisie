import cv2
import numpy as np
import io
import sys
import io_utils
import csv

scale = 1.0

def draw_point (event, x, y, flags, corners):
    if event == cv2.EVENT_LBUTTONDOWN:
        corners.append(int(x*scale))
        corners.append(int(y*scale))
        cv2.circle(img_scaled, (x,y), 3, (0,255,0), thickness=-1)
        cv2.imshow(imname, img_scaled)


with open("corners_8.csv", mode = "w") as pandc:
    corner_writer = csv.writer(pandc, delimiter=";", quotechar="\"", quoting=csv.QUOTE_MINIMAL)

    for path, img in io_utils.imread_folder("images\zalen\zaal_8"):
        path2 = path.split("\\")
        imname = path2[len(path2) - 1]
        print(imname)
        height, width, depth = img.shape
        if (height > 700):
            scale = height/700
            img_scaled = cv2.resize(img, (int(width / scale), int(height / scale)))  # width height
        else:
            scale = 1
        print(scale)
        cv2.namedWindow(imname)
        corners = []
        cv2.setMouseCallback(imname, draw_point, corners)
        cv2.imshow(imname, img_scaled)
        cv2.moveWindow(imname, 100, 0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(corners)
        print(len(corners))
        if(len(corners) >= 7):
            print("writing")
            corner_writer.writerow([imname, corners[0], corners[1], corners[2], corners[3],
                                corners[4], corners[5], corners[6], corners[7]])

