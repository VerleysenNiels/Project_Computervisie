import cv2
import numpy as np
import io
import sys
import io_utils
import csv

"""
    Run this with the right settings for the folder to read and the csv to write
    and it will show every image in turn. When it shows an image, click the four corners
    (only the first four are recorded, so do it right) and then press any button to go
    to the next image. Make sure you exit clean, or nothing will be saved in the csv.
    Also, don't overwrite existing files.
    
    Corner coordinates are in the images scaled to 1080.
"""

scale = 1.0

def draw_point (event, x, y, flags, corners):
    if event == cv2.EVENT_LBUTTONDOWN:
        corners.append(int(x*scale))
        corners.append(int(y*scale))
        cv2.circle(img_scaled, (x,y), 3, (0,255,0), thickness=-1)
        cv2.imshow(imname, img_scaled)


with open("corners_11.csv", mode = "w") as pandc:
    corner_writer = csv.writer(pandc, delimiter=";", quotechar="\"", quoting=csv.QUOTE_MINIMAL)

    for path, img in io_utils.imread_folder("images\zalen\zaal_11"):
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

