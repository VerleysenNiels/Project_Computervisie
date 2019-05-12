import cv2
import io_utils
import csv
import perspective
import math

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
    a1 = (line1[1, 1] - line1[0, 1]) / (line1[1, 0] - line1[0, 0])
    a2 = (line2[1, 1] - line2[0, 1]) / (line2[1, 0] - line2[0, 0])
    phi = math.atan((a1 - a2)/(1 - (a1*a2)))
    return phi

def calculate_area (pts):
    a = euclidian_dist(pts[2], pts[3])
    b = euclidian_dist(pts[3], pts[0])
    c = euclidian_dist(pts[0], pts[1])
    d = euclidian_dist(pts[1], pts[2])
    t = 0.5 * (a + b + c + d)
    angle1 = angle_betw_lines((pts[2], pts[3]), (pts[3], pts[0]))
    angle2 = angle_betw_lines((pts[0], pts[1]), (pts[1], pts[2]))
    area = math.sqrt(((t - a) * (t - b) * (t - c) * (t - d))
                     - (a * b * c * d * ((math.cos((angle1 + angle2)/2)) ** 2)))
    return area

def calculate_intersection (pts1, pts2):
    # = intersections(p11, p12, p21, p22)
    intersection = calculate_area(pts1)
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

