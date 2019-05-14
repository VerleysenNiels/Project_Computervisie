import cv2
import io_utils
import math_utils
import csv
import perspective
import math
import numpy as np

"""
    Run this with the right settings for the folder to read and the csv to write
    and it will show every image in turn. When it shows an image, click the four corners
    (only the first four are recorded, so do it right) and then press any button to go
    to the next image. Make sure you exit clean, or nothing will be saved in the csv.
    Also, don't overwrite existing files.
    
    Corner coordinates are in the images scaled to 1080
    (had to scale them down to 700 because I have a shitty screen - Ralph).
"""

scale = 1.0

def draw_point (event, x, y, flags, corners):
    if event == cv2.EVENT_LBUTTONDOWN:
        corners.append(int(x*scale))
        corners.append(int(y*scale))
        cv2.circle(img_scaled, (x,y), 3, (0,255,0), thickness=-1)
        cv2.imshow(imname, img_scaled)

def euclidian_dist(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def angle_betw_lines (line1, line2):
    # equation as "y = ax + b"
    # a = (y2 - y1)/(x2 - x1)
    dx1 = (line1[1, 0] - line1[0, 0])
    dx2 = (line2[1, 0] - line2[0, 0])
    dy1 = (line1[1, 1] - line1[0, 1])
    dy2 = (line2[1, 1] - line2[0, 1])
    theta1 = math.pi/2
    theta2 = math.pi/2
    if dx1 != 0:
        theta1 = math.atan(dy1 / dx1)
    if dx2 != 0:
        theta2 = math.atan(dy2 / dx2)
    phi = theta1 - theta2
    while phi<0:
        phi += math.pi
    return phi

def calculate_area (pts):
    pts = perspective.order_points(pts)
    a = euclidian_dist(pts[2], pts[3])
    b = euclidian_dist(pts[3], pts[0])
    c = euclidian_dist(pts[0], pts[1])
    d = euclidian_dist(pts[1], pts[2])
    t = 0.5 * (a + b + c + d)
    angle1 = angle_betw_lines(np.array([pts[2], pts[3]]), np.array([pts[3], pts[0]]))
    angle2 = angle_betw_lines(np.array([pts[0], pts[1]]), np.array([pts[1], pts[2]]))
    area = math.sqrt(((t - a) * (t - b) * (t - c) * (t - d))
                     - (a * b * c * d * ((math.cos((angle1 + angle2)/2)) ** 2)))
    return area

def calculate_intersection(pts1, pts2):
    pts3 = pts1
    #first check if either corner lies completely inside of the other (that's the two first ifs
    #if not, check which lies most inside to get the inside intersection
    if pts1[0][0] > pts2[0][0] and pts1[0][1] > pts2[0][1]:
        pts3[0] = pts1[0]
    elif pts1[0][0] < pts1[0][0] and pts1[0][1] < pts1[0][1]:
        pts3[0] = pts2[0]
    elif pts1[0][0] > pts2[0][0]:
        pts3[0] = math_utils.intersections(np.array([pts1[0][0], pts1[0][1], pts1[3][0], pts1[3][1]]),
                                           np.array([pts2[0][0], pts2[0][1], pts2[1][0], pts2[1][1]]))
    else:
        pts3[0] = math_utils.intersections(np.array([pts1[0][0], pts1[0][1], pts1[1][0], pts1[1][1]]),
                                           np.array([pts2[0][0], pts2[0][1], pts2[3][0], pts2[3][1]]))

    if pts1[1][0] < pts2[1][0] and pts1[1][1] > pts2[1][1]:
        pts3[1] = pts1[1]
    elif pts1[1][0] > pts2[1][0] and pts1[1][1] < pts2[1][1]:
        pts3[1] = pts2[1]
    elif pts1[1][0] < pts2[1][0]:
        pts3[1] = math_utils.intersections(np.array([pts1[1][0], pts1[1][1], pts1[2][0], pts1[2][1]]),
                                           np.array([pts2[0][0], pts2[0][1], pts2[1][0], pts2[1][1]]))
    else:
        pts3[1] = math_utils.intersections(np.array([pts1[0][0], pts1[0][1], pts1[1][0], pts1[1][1]]),
                                           np.array([pts2[1][0], pts2[1][1], pts2[2][0], pts2[2][1]]))

    if pts1[2][0] < pts2[2][0] and pts1[2][1] < pts2[2][1]:
        pts3[2] = pts1[2]
    elif pts1[2][0] > pts2[2][0] and pts1[2][1] > pts2[2][1]:
        pts3[2] = pts2[2]
    elif pts1[2][0] < pts2[2][0]:
        pts3[2] = math_utils.intersections(np.array([pts1[1][0], pts1[1][1], pts1[2][0], pts1[2][1]]),
                                           np.array([pts2[2][0], pts2[2][1], pts2[3][0], pts2[3][1]]))
    else:
        pts3[2] = math_utils.intersections(np.array([pts1[2][0], pts1[2][1], pts1[3][0], pts1[3][1]]),
                                           np.array([pts2[1][0], pts2[1][1], pts2[2][0], pts2[2][1]]))

    if pts1[3][0] > pts2[3][0] and pts1[3][1] < pts2[3][1]:
        pts3[3] = pts1[3]
    elif pts1[3][0] < pts2[3][0] and pts1[3][1] > pts2[3][1]:
        pts3[3] = pts2[3]
    elif pts1[3][0] > pts2[0][0]:
        pts3[3] = math_utils.intersections(np.array([pts1[3][0], pts1[3][1], pts1[0][0], pts1[0][1]]),
                                           np.array([pts2[2][0], pts2[2][1], pts2[3][0], pts2[3][1]]))
    else:
        pts3[3] = math_utils.intersections(np.array([pts1[2][0], pts1[2][1], pts1[3][0], pts1[3][1]]),
                                           np.array([pts2[0][0], pts2[0][1], pts2[3][0], pts2[3][1]]))

    intersection = calculate_area(pts3)
    return intersection


def calculate_union (pts1, pts2):
    union = calculate_area(pts1) + calculate_area(pts2) - calculate_intersection(pts1, pts2)
    return union

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

